// Package kvmirror keeps a copy of a vLLM engine's GPU prefix cache from its KV cache events
// (--kv-events-config, ZMQ PUB). With the token IDs of an incoming prompt it answers
// "how many of these tokens are cached right now", which is what admission prices.
//
// Wire format (vllm/distributed/kv_events.py): multipart [topic, seq uint64 BE, payload];
// payload is msgpack [ts, [event...], dp_rank?], each event a map tagged by "type":
//
//	BlockStored   block_hashes, parent_block_hash, token_ids, block_size, extra_keys,
//	              group_idx, kv_cache_spec_kind
//	BlockRemoved  block_hashes, group_idx
//	AllBlocksCleared
//
// Engine block hashes are opaque (sha256 over pickled keys), so the mirror keys blocks by
// its own chain hash of token IDs: key(b) = H(key(parent), tokens(b)). An event's parent
// hash is mapped to a key through blocks the mirror has already seen; a block whose
// parent is unknown (stored before the proxy subscribed) is skipped, which only makes the
// mirror under-report hits. Blocks with extra_keys (images, LoRA, cache salt) hash with
// those keys too, so token IDs alone never match them. Only full blocks are tracked: a
// prompt's keys are built at one block size and match only blocks stored at that size.
//
// Hybrid models (measured on Qwen3.8 Flash-Next, 2026-10-07): attention groups store a block as soon
// as it is prefilled; each recurrent (mamba) group keeps a state checkpoint at about one
// block per request. vLLM's hybrid coordinator (kv_cache_coordinator.py
// find_longest_cache_hit) takes the contiguous attention blocks, drops the last one when
// speculative decoding is on (the EAGLE drop; MTP flags every group), then
// the newest block at or below that where every mamba group holds a checkpoint.
package kvmirror

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

type BlockKey [16]byte

type mirrorBlock struct {
	key    BlockKey
	groups uint64 // bit per KV cache group holding at least one copy
	// copies per group: vLLM can hold several physical blocks with one hash (a hybrid
	// model recomputes attention blocks past its last recurrent checkpoint), and evicting
	// one copy publishes BlockRemoved for the hash while another stays cached.
	copies map[uint]int32
}

type Mirror struct {
	mu        sync.Mutex
	blockSize int                     // set by SetBlockSize, else learned from the first stored block
	byEngine  map[uint64]*mirrorBlock // engine block hash -> block
	byKey     map[BlockKey]*mirrorBlock
	attnMask  uint64 // groups that are attention (need a contiguous prefix)
	mambaMask uint64 // groups that are recurrent state (need the hit block only)
	lastSeq   int64
	// After a replay, live delivery repeats the batches the replay already applied, once,
	// in order, and at once (they queued while the replay ran): for overlapWindow after
	// the replay, a sequence from firstSeq up to replayedTo, above liveSeq, is a
	// duplicate. Any other sequence at or below lastSeq means the engine restarted.
	firstSeq, replayedTo, liveSeq int64
	replayedAt                    time.Time
	lastEvent                     time.Time
	connected                     bool
	// generation counts resets: subscriptions, engine restarts, AllBlocksCleared.
	generation uint64
	sizeWarned bool // a block of another size than blockSize was logged
	// dropLast applies the EAGLE last-block drop (speculative decoding on).
	dropLast bool
	stats    Stats
}

// PromptKeys are a prompt's mirror keys: one chained key per full block, at the block
// size they were built with. Keys built at another size than the mirror's match nothing.
type PromptKeys struct {
	BlockSize int
	Blocks    []BlockKey
}

func (k PromptKeys) Empty() bool { return len(k.Blocks) == 0 }

// NewPromptKeys computes a prompt's keys at the mirror's block size (none while it is
// unknown).
func (m *Mirror) NewPromptKeys(prompt []int64) PromptKeys {
	bs := m.BlockSize()
	return PromptKeys{BlockSize: bs, Blocks: PromptBlockKeys(prompt, bs)}
}

// SetBlockSize sets the engine's block size (from its metrics), so prompt keys are built
// at the right size before the first event shows it.
func (m *Mirror) SetBlockSize(blockSize int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if blockSize > 0 {
		m.blockSize = blockSize
	}
}

type Stats struct {
	Stored, Removed, Cleared, Restarts, SkippedOrphan, SkippedEmpty, Batches, Duplicates, SeqGaps uint64
}

func New() *Mirror {
	m := &Mirror{dropLast: true}
	m.resetLocked()
	return m
}

// errSeqGap ends a subscription so it restarts with a replay.
var errSeqGap = fmt.Errorf("kv events sequence gap")

var rootKey BlockKey

func chainKey(parent BlockKey, tokens []int64, extra []byte) BlockKey {
	h := sha256.New()
	h.Write(parent[:])
	var b [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(b[:], uint64(t))
		h.Write(b[:])
	}
	if len(extra) > 0 {
		h.Write([]byte{0xff})
		h.Write(extra)
	}
	var k BlockKey
	copy(k[:], h.Sum(nil))
	return k
}

type Event struct {
	Type            string  `msgpack:"type"`
	BlockHashes     []any   `msgpack:"block_hashes"`
	ParentBlockHash any     `msgpack:"parent_block_hash"`
	TokenIDs        []int64 `msgpack:"token_ids"`
	BlockSize       int     `msgpack:"block_size"`
	ExtraKeys       []any   `msgpack:"extra_keys"`
	GroupIdx        *int    `msgpack:"group_idx"`
	Kind            string  `msgpack:"kv_cache_spec_kind"`
	Medium          *string `msgpack:"medium"`
}

func hashToUint(v any) (uint64, bool) {
	switch x := v.(type) {
	case uint64:
		return x, true
	case int64:
		return uint64(x), true
	case int:
		return uint64(x), true
	case uint32:
		return uint64(x), true
	case int32:
		return uint64(x), true
	case int8:
		return uint64(x), true
	case uint8:
		return uint64(x), true
	case int16:
		return uint64(x), true
	case uint16:
		return uint64(x), true
	case []byte:
		var u uint64
		for _, c := range x {
			u = u<<8 | uint64(c)
		}
		return u, true
	}
	return 0, false
}

// decodeBatch parses one payload into events.
func decodeBatch(payload []byte) ([]Event, error) {
	var batch []msgpack.RawMessage
	if err := msgpack.Unmarshal(payload, &batch); err != nil {
		return nil, fmt.Errorf("batch: %w", err)
	}
	if len(batch) < 2 {
		return nil, fmt.Errorf("batch has %d fields", len(batch))
	}
	var events []Event
	if err := msgpack.Unmarshal(batch[1], &events); err != nil {
		return nil, fmt.Errorf("events: %w", err)
	}
	return events, nil
}

// Apply ingests one batch. seq is the publisher's sequence number (-1: unknown). Live
// delivery repeating what a replay applied is skipped; any other sequence that goes
// backwards means the engine restarted, so the mirror starts over. It returns false on a
// gap: batches were lost and the mirror can no longer be trusted until it resyncs.
func (m *Mirror) Apply(seq int64, events []Event) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if seq >= 0 && m.lastSeq >= 0 {
		if seq >= m.firstSeq && seq <= m.replayedTo && seq > m.liveSeq && time.Since(m.replayedAt) < overlapWindow {
			m.liveSeq = seq
			m.stats.Duplicates++
			return true
		}
		m.replayedTo = -1
		switch {
		case seq <= m.lastSeq:
			log.Printf("level=warn msg=\"kv events sequence went back: engine restarted, mirror cleared\" seq=%d last=%d", seq, m.lastSeq)
			m.resetLocked()
			m.stats.Restarts++
		case seq > m.lastSeq+1:
			m.stats.SeqGaps++
			m.lastSeq = seq
			return false
		}
	}
	if seq >= 0 && m.firstSeq < 0 {
		m.firstSeq = seq
	}
	if seq >= 0 {
		m.lastSeq = seq
	}
	m.lastEvent = time.Now()
	m.stats.Batches++
	for i := range events {
		m.applyLocked(&events[i])
	}
	return true
}

// resetLocked forgets every block and what the groups are (a restarted engine may serve
// another model); the block size set from metrics stays.
func (m *Mirror) resetLocked() {
	m.generation++
	m.byEngine = map[uint64]*mirrorBlock{}
	m.byKey = map[BlockKey]*mirrorBlock{}
	m.attnMask, m.mambaMask = 0, 0
	m.firstSeq, m.lastSeq, m.replayedTo, m.liveSeq = -1, -1, -1, -1
}

func (m *Mirror) applyLocked(e *Event) {
	if e.Medium != nil && *e.Medium != "" && *e.Medium != "GPU" {
		return
	}
	switch e.Type {
	case "AllBlocksCleared":
		m.generation++
		m.byEngine = map[uint64]*mirrorBlock{}
		m.byKey = map[BlockKey]*mirrorBlock{}
		m.stats.Cleared++
	case "BlockRemoved":
		if e.GroupIdx == nil {
			return
		}
		g := uint(*e.GroupIdx)
		for _, h := range e.BlockHashes {
			u, ok := hashToUint(h)
			b := m.byEngine[u]
			if !ok || b == nil {
				continue
			}
			m.stats.Removed++
			if b.copies[g] > 1 {
				b.copies[g]--
				continue
			}
			delete(b.copies, g)
			b.groups &^= 1 << g
			if b.groups == 0 {
				delete(m.byEngine, u)
				if m.byKey[b.key] == b {
					delete(m.byKey, b.key)
				}
			}
		}
	case "BlockStored":
		if e.GroupIdx == nil || e.BlockSize <= 0 {
			return
		}
		g := uint(*e.GroupIdx)
		bit := uint64(1) << g
		if e.Kind == "mamba" {
			m.mambaMask |= bit
		} else {
			m.attnMask |= bit
		}
		if m.blockSize == 0 {
			m.blockSize = e.BlockSize
		} else if e.BlockSize != m.blockSize && !m.sizeWarned {
			// Keys are built at the engine's block size; blocks of another size never match.
			m.sizeWarned = true
			log.Printf("level=warn msg=\"kv events block size differs from the engine's: those blocks never match\" event_block_size=%d block_size=%d group=%d", e.BlockSize, m.blockSize, g)
		}
		if len(e.BlockHashes) == 0 {
			m.stats.SkippedEmpty++
			return
		}
		parent := rootKey
		if e.ParentBlockHash != nil {
			pu, ok := hashToUint(e.ParentBlockHash)
			pb := m.byEngine[pu]
			if !ok || pb == nil {
				// The rest of this event chains from the unknown parent; the batch's other
				// events are applied as usual.
				m.stats.SkippedOrphan++
				return
			}
			parent = pb.key
		}
		bs := e.BlockSize
		for i, h := range e.BlockHashes {
			if (i+1)*bs > len(e.TokenIDs) {
				return
			}
			var extra []byte
			if i < len(e.ExtraKeys) && e.ExtraKeys[i] != nil {
				extra, _ = msgpack.Marshal(e.ExtraKeys[i])
			}
			key := chainKey(parent, e.TokenIDs[i*bs:(i+1)*bs], extra)
			parent = key
			u, ok := hashToUint(h)
			if !ok {
				continue // its key still chains the blocks after it
			}
			b := m.byKey[key]
			if b == nil {
				b = &mirrorBlock{key: key, copies: map[uint]int32{}}
				m.byKey[key] = b
			}
			m.byEngine[u] = b
			b.copies[g]++
			b.groups |= bit
			m.stats.Stored++
		}
	}
}

// PromptBlockKeys chains the mirror's block keys over a prompt's full blocks, excluding
// the block holding the last token (vLLM always computes that one). Computed once per
// request, so rescoring a waiter costs map lookups only.
func PromptBlockKeys(prompt []int64, blockSize int) []BlockKey {
	if blockSize <= 0 || len(prompt) < 2 {
		return nil
	}
	n := (len(prompt) - 1) / blockSize
	keys := make([]BlockKey, n)
	k := rootKey
	for i := 0; i < n; i++ {
		k = chainKey(k, prompt[i*blockSize:(i+1)*blockSize], nil)
		keys[i] = k
	}
	return keys
}

// Cached returns how many leading tokens of a prompt the engine would serve from its
// prefix cache now.
func (m *Mirror) Cached(k PromptKeys) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if k.BlockSize <= 0 || k.BlockSize != m.blockSize || m.attnMask == 0 {
		return 0
	}
	contiguous := 0
	for _, key := range k.Blocks {
		b := m.byKey[key]
		if b == nil || b.groups&m.attnMask != m.attnMask {
			break
		}
		contiguous++
	}
	if m.dropLast {
		contiguous--
	}
	for n := contiguous; n > 0; n-- {
		if m.mambaMask == 0 || m.byKey[k.Blocks[n-1]].groups&m.mambaMask == m.mambaMask {
			return n * k.BlockSize
		}
	}
	return 0
}

// BlockSize is the engine's block size once set or shown by an event, else 0.
func (m *Mirror) BlockSize() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.blockSize
}

// CachedTokens returns how many leading tokens of prompt the engine would serve from
// its prefix cache now.
func (m *Mirror) CachedTokens(prompt []int64) int {
	return m.Cached(m.NewPromptKeys(prompt))
}

type Status struct {
	Connected     bool    `json:"connected"`
	Blocks        int     `json:"blocks"`
	BlockSize     int     `json:"block_size"`
	LastEventAgeS float64 `json:"last_event_age_s"`
	Stats         Stats   `json:"stats"`
}

func (m *Mirror) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := Status{Connected: m.connected, Blocks: len(m.byKey), BlockSize: m.blockSize, Stats: m.stats}
	if !m.lastEvent.IsZero() {
		s.LastEventAgeS = time.Since(m.lastEvent).Seconds()
	}
	return s
}

// Healthy reports whether the mirror is subscribed and its replay completed. An idle
// engine sends no events, so silence alone is not a failure.
func (m *Mirror) Healthy() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.connected
}

// SetConnected marks the mirror as subscribed (or not). The subscription loop (Run) calls it;
// it is exported for programs that feed events through Apply themselves, and for tests.
func (m *Mirror) SetConnected(v bool) {
	m.mu.Lock()
	m.connected = v
	m.mu.Unlock()
}

// Run subscribes to endpoint and applies batches until ctx ends, reconnecting on error.
// With a replay endpoint, each subscription starts from an empty mirror rebuilt from the
// publisher's replay buffer, and the mirror is healthy only once that replay completed.
func (m *Mirror) Run(ctx context.Context, endpoint, replay string) {
	for ctx.Err() == nil {
		err := m.subscribe(ctx, endpoint, replay)
		m.SetConnected(false)
		if ctx.Err() != nil {
			return
		}
		log.Printf("level=warn msg=\"kv events subscription ended\" endpoint=%q err=%v", endpoint, err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}

func (m *Mirror) subscribe(ctx context.Context, endpoint, replay string) error {
	sub := zmq4.NewSub(ctx, zmq4.WithAutomaticReconnect(false), zmq4.WithDialerRetry(time.Second))
	defer sub.Close()
	if err := sub.Dial(endpoint); err != nil {
		return err
	}
	if err := sub.SetOption(zmq4.OptionSubscribe, ""); err != nil {
		return err
	}
	m.mu.Lock()
	m.resetLocked()
	m.mu.Unlock()
	// Live delivery starts now and queues while the replay buffer (the publisher's last
	// buffer_steps batches) rebuilds what was stored before; the queued batches the replay
	// already covered are then skipped.
	if replay != "" {
		n, err := m.replay(ctx, replay)
		if err != nil {
			return fmt.Errorf("replay: %w", err)
		}
		log.Printf("level=info msg=\"kv events replayed\" batches=%d blocks=%d", n, m.Status().Blocks)
	}
	m.mu.Lock()
	m.endReplayLocked()
	m.connected = true
	m.mu.Unlock()
	log.Printf("level=info msg=\"kv events subscribed\" endpoint=%q", endpoint)
	for {
		msg, err := sub.Recv()
		if err != nil {
			return err
		}
		seq, events, ok := parseFrames(msg.Frames)
		if !ok {
			continue
		}
		if !m.Apply(seq, events) {
			return errSeqGap
		}
	}
}

// parseFrames reads [topic, seq, payload] (or [payload]) into a batch.
func parseFrames(frames [][]byte) (int64, []Event, bool) {
	if len(frames) == 0 {
		return -1, nil, false
	}
	payload := frames[len(frames)-1]
	seq := int64(-1)
	if len(frames) >= 2 && len(frames[len(frames)-2]) == 8 {
		seq = int64(binary.BigEndian.Uint64(frames[len(frames)-2]))
	}
	events, err := decodeBatch(payload)
	if err != nil {
		log.Printf("level=warn msg=\"kv events decode\" err=%v", err)
		return seq, nil, false
	}
	return seq, events, true
}

// replay rebuilds recent state from the publisher's replay buffer (its last
// buffer_steps batches). The publisher's ROUTER queues at most ~1,000 messages for a
// reader and drops the rest silently, so a replay is read in rounds: each takes the
// batches that arrive in sequence, stops at the first gap or a quiet second, and asks
// again from the next missing sequence, until the end-of-sequence marker (seq -1). A
// replay that does not reach the marker is an error.
func (m *Mirror) replay(ctx context.Context, endpoint string) (int, error) {
	total := 0
	from := int64(0)
	for round := 0; round < 64; round++ {
		n, next, done, err := m.replayRound(ctx, endpoint, from)
		total += n
		if err != nil || done {
			return total, err
		}
		if n == 0 {
			return total, fmt.Errorf("replay round from %d returned nothing", from)
		}
		from = next
	}
	return total, fmt.Errorf("replay did not finish in 64 rounds")
}

func (m *Mirror) replayRound(ctx context.Context, endpoint string, from int64) (n int, next int64, done bool, err error) {
	rctx, cancel := context.WithCancel(ctx)
	defer cancel()
	d := zmq4.NewDealer(rctx)
	defer d.Close()
	if err := d.Dial(endpoint); err != nil {
		return 0, from, false, err
	}
	// The connection handshake completes asynchronously; a request sent before it is lost.
	time.Sleep(300 * time.Millisecond)
	start := make([]byte, 8)
	binary.BigEndian.PutUint64(start, uint64(from))
	if err := d.Send(zmq4.NewMsgFrom([]byte{}, start)); err != nil {
		return 0, from, false, err
	}
	msgs := make(chan zmq4.Msg, 64)
	go func() {
		defer close(msgs)
		for {
			msg, err := d.Recv()
			if err != nil {
				return
			}
			select {
			case msgs <- msg:
			case <-rctx.Done():
				return
			}
		}
	}()
	next = from
	idle := time.NewTimer(replayFirstWait)
	defer idle.Stop()
	for {
		select {
		case <-ctx.Done():
			return n, next, false, ctx.Err()
		case <-idle.C:
			return n, next, false, nil
		case msg, ok := <-msgs:
			if !ok {
				return n, next, false, nil
			}
			f := msg.Frames
			if len(f) > 0 && len(f[0]) == 0 {
				f = f[1:] // delimiter
			}
			if len(f) >= 2 && len(f[len(f)-2]) == 8 && int64(binary.BigEndian.Uint64(f[len(f)-2])) == -1 {
				return n, next, true, nil
			}
			seq, events, ok := parseFrames(f)
			if !ok {
				continue
			}
			if seq != next && next != 0 {
				return n, next, false, nil // the publisher dropped batches: ask again from here
			}
			if seq < next {
				continue
			}
			m.Apply(seq, events)
			n++
			next = seq + 1
			idle.Reset(time.Second)
		}
	}
}

// ReplayEndpoint is the publisher's replay service, expected on the next port
// (--kv-events-config replay_endpoint).
func ReplayEndpoint(endpoint string) string {
	i := strings.LastIndexByte(endpoint, ':')
	port, err := strconv.Atoi(endpoint[i+1:])
	if i < 0 || err != nil {
		return ""
	}
	return endpoint[:i+1] + strconv.Itoa(port+1)
}

// replayFirstWait bounds the wait for a replay round's first batch.
var replayFirstWait = 5 * time.Second

// endReplayLocked opens the window in which live delivery may repeat what the replay
// applied.
func (m *Mirror) endReplayLocked() {
	m.replayedTo, m.liveSeq, m.replayedAt = m.lastSeq, -1, time.Now()
}

// overlapWindow bounds how long after a replay live batches can still be repeats of it.
// They queued while the replay ran and arrive at once; later, a sequence going back is a
// restart even inside the replayed range (an idle engine restarting starts at 0).
var overlapWindow = 5 * time.Second

// Generation changes whenever the mirror forgets what it held: a (re)subscription, an
// engine restart, or the engine clearing its cache. A block the mirror does not see now
// was seen and lost if the generation is the same as when it should have been stored.
func (m *Mirror) Generation() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.generation
}
