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
// those keys too, so token IDs alone never match them.
//
// Hybrid models (measured on Qwen3.8 Flash-Next, 2026-10-07): attention groups store a block as soon
// as it is prefilled; each recurrent (mamba) group keeps a state checkpoint at about one
// block per request. vLLM's hybrid coordinator (kv_cache_coordinator.py
// find_longest_cache_hit) takes the contiguous attention blocks, drops the last one when
// speculative decoding is on (the EAGLE drop; MTP flags every group), then
// the newest block at or below that where every mamba group holds a checkpoint.
//
// Fine-grained hits (--prefix-match-unit, 2026-10-08): the engine also hashes every
// unit-token span (64) and, at each prompt's tail, stores a recurrent checkpoint ending at
// position p (event block_size = unit) plus an attention entry for the span [p, p+unit)
// whose parent is that checkpoint. Their own parents are spans the engine never
// announces, so they cannot be chained: the mirror indexes them by the hash of their own
// tokens instead. A prompt hits at p when every full block before p's block is cached,
// a checkpoint entry matches its tokens [p-unit, p), and an attention entry matching
// [p, p+unit) names that checkpoint as its parent (the last-span drop leaves p).
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
	blockSize int
	byEngine  map[uint64]*mirrorBlock // engine block hash -> block
	byKey     map[BlockKey]*mirrorBlock
	attnMask  uint64 // groups that are attention (need a contiguous prefix)
	mambaMask uint64 // groups that are recurrent state (need the hit block only)
	lastSeq   int64
	lastEvent time.Time
	connected bool
	// dropLast applies the EAGLE last-block drop (speculative decoding on).
	dropLast bool
	stats    Stats

	// epoch counts subscriptions; state seen under an older one may be missing now
	// without having been evicted (the replay window is finite).
	epoch uint64

	unit          int // prefix-match unit; < blockSize when fine-grained hits are on
	fineByEngine  map[uint64]*fineEntry
	fineByContent map[uint64][]*fineEntry
}

// fineEntry is one announced unit-sized span: a tail checkpoint or the attention span
// after it.
type fineEntry struct {
	engine, parent uint64
	content        uint64 // unitHash of its tokens
	groups         uint64
	copies         map[uint]int32
}

// PromptKeys are a prompt's precomputed mirror keys: chained full-block keys, and the
// content hash of every full unit-token span.
type PromptKeys struct {
	Blocks []BlockKey
	Units  []uint64
	Unit   int
}

func (k PromptKeys) Empty() bool { return len(k.Blocks) == 0 && len(k.Units) == 0 }

func unitHash(tokens []int64) uint64 {
	h := sha256.New()
	var b [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(b[:], uint64(t))
		h.Write(b[:])
	}
	return binary.LittleEndian.Uint64(h.Sum(nil))
}

// NewPromptKeys computes a prompt's keys for the current geometry.
func (m *Mirror) NewPromptKeys(prompt []int64) PromptKeys {
	m.mu.Lock()
	bs, unit := m.blockSize, m.unit
	m.mu.Unlock()
	k := PromptKeys{Blocks: PromptBlockKeys(prompt, bs)}
	if unit > 0 && unit < bs {
		k.Unit = unit
		k.Units = make([]uint64, len(prompt)/unit)
		for i := range k.Units {
			k.Units[i] = unitHash(prompt[i*unit : (i+1)*unit])
		}
	}
	return k
}

// SetGeometry presets the block size and prefix-match unit (from the engine's metrics),
// so events are classified correctly from the first one.
func (m *Mirror) SetGeometry(blockSize, unit int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.blockSize = max(m.blockSize, blockSize)
	if unit > 0 {
		m.unit = unit
	}
}

type Stats struct {
	Stored, Removed, Cleared, SkippedOrphan, SkippedEmpty, Batches, SeqGaps uint64
}

func New() *Mirror {
	return &Mirror{byEngine: map[uint64]*mirrorBlock{}, byKey: map[BlockKey]*mirrorBlock{}, lastSeq: -1, dropLast: true,
		fineByEngine: map[uint64]*fineEntry{}, fineByContent: map[uint64][]*fineEntry{}}
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

// Apply ingests one batch. seq is the publisher's sequence number (-1: unknown). A
// batch already applied (replay overlapping live delivery) is skipped; a much lower
// number means the engine restarted, so the mirror starts over. It returns false on a
// gap: batches were lost and the mirror can no longer be trusted until it resyncs.
func (m *Mirror) Apply(seq int64, events []Event) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if seq >= 0 && m.lastSeq >= 0 {
		switch {
		case seq <= m.lastSeq && m.lastSeq-seq < replayOverlap:
			return true // duplicate
		case seq <= m.lastSeq:
			m.clearLocked() // engine restarted
		case seq > m.lastSeq+1:
			m.stats.SeqGaps++
			m.lastSeq = seq
			return false
		}
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

// replayOverlap bounds how far behind the last applied sequence a batch may be and still
// count as a duplicate rather than an engine restart.
const replayOverlap = 4096

func (m *Mirror) clearLocked() {
	m.byEngine = map[uint64]*mirrorBlock{}
	m.byKey = map[BlockKey]*mirrorBlock{}
	m.fineByEngine = map[uint64]*fineEntry{}
	m.fineByContent = map[uint64][]*fineEntry{}
	m.stats.Cleared++
}

func (m *Mirror) applyLocked(e *Event) {
	if e.Medium != nil && *e.Medium != "" && *e.Medium != "GPU" {
		return
	}
	switch e.Type {
	case "AllBlocksCleared":
		m.clearLocked()
	case "BlockRemoved":
		if e.GroupIdx == nil {
			return
		}
		bit := uint64(1) << uint(*e.GroupIdx)
		for _, h := range e.BlockHashes {
			u, ok := hashToUint(h)
			if !ok {
				continue
			}
			b := m.byEngine[u]
			if b == nil {
				m.removeFineLocked(u, uint(*e.GroupIdx))
				continue
			}
			g := uint(*e.GroupIdx)
			if b.copies[g] > 1 {
				b.copies[g]--
				m.stats.Removed++
				continue
			}
			delete(b.copies, g)
			b.groups &^= bit
			m.stats.Removed++
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
		m.blockSize = max(m.blockSize, e.BlockSize)
		if len(e.BlockHashes) == 0 {
			m.stats.SkippedEmpty++
			return
		}
		if e.BlockSize < m.blockSize {
			m.storeFineLocked(e, g, bit)
			return
		}
		parent := rootKey
		if e.ParentBlockHash != nil {
			pu, ok := hashToUint(e.ParentBlockHash)
			pb := m.byEngine[pu]
			if !ok || pb == nil {
				m.stats.SkippedOrphan++
				return
			}
			parent = pb.key
		}
		bs := e.BlockSize
		for i, h := range e.BlockHashes {
			u, ok := hashToUint(h)
			if !ok || (i+1)*bs > len(e.TokenIDs) {
				return
			}
			var extra []byte
			if i < len(e.ExtraKeys) && e.ExtraKeys[i] != nil {
				extra, _ = msgpack.Marshal(e.ExtraKeys[i])
			}
			key := chainKey(parent, e.TokenIDs[i*bs:(i+1)*bs], extra)
			b := m.byEngine[u]
			if b == nil {
				if kb := m.byKey[key]; kb != nil {
					b = kb
				} else {
					b = &mirrorBlock{key: key}
					m.byKey[key] = b
				}
				m.byEngine[u] = b
			}
			if b.copies == nil {
				b.copies = map[uint]int32{}
			}
			b.copies[g]++
			b.groups |= bit
			m.stats.Stored++
			parent = key
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

// CachedBlockKeys is CachedTokens over precomputed keys.
func (m *Mirror) CachedBlockKeys(keys []BlockKey) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.blockSize <= 0 || m.attnMask == 0 {
		return 0
	}
	contiguous := 0
	for _, k := range keys {
		b := m.byKey[k]
		if b == nil || b.groups&m.attnMask != m.attnMask {
			break
		}
		contiguous++
	}
	if m.dropLast {
		contiguous--
	}
	for n := contiguous; n > 0; n-- {
		if m.mambaMask == 0 || m.byKey[keys[n-1]].groups&m.mambaMask == m.mambaMask {
			return n * m.blockSize
		}
	}
	return 0
}

// Epoch identifies the current subscription.
func (m *Mirror) Epoch() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.epoch
}

// BlockSize is the engine's block size once an event has shown it, else 0.
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

// Healthy reports whether the mirror is subscribed. An idle engine sends no events, so
// silence alone is not a failure.
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
	m.clearLocked()
	m.lastSeq = -1
	m.epoch++
	m.mu.Unlock()
	// Live delivery starts now; the replay buffer (the publisher's last buffer_steps
	// batches) rebuilds what was stored before, and overlapping batches are skipped.
	if replay != "" {
		if n, err := m.replay(ctx, replay); err != nil {
			log.Printf("level=warn msg=\"kv events replay failed\" err=%v", err)
		} else {
			log.Printf("level=info msg=\"kv events replayed\" batches=%d blocks=%d", n, m.Status().Blocks)
		}
	}
	m.SetConnected(true)
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
// again from the next missing sequence, until the end-of-sequence marker (seq -1).
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
	idle := time.NewTimer(5 * time.Second) // the first batch can take a moment
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

func (m *Mirror) storeFineLocked(e *Event, g uint, bit uint64) {
	if m.unit == 0 {
		m.unit = e.BlockSize
	}
	bs := e.BlockSize
	parent, _ := hashToUint(e.ParentBlockHash)
	for i, h := range e.BlockHashes {
		u, ok := hashToUint(h)
		if !ok || (i+1)*bs > len(e.TokenIDs) {
			return
		}
		if i < len(e.ExtraKeys) && e.ExtraKeys[i] != nil {
			return // media or salt: tokens alone never match it
		}
		f := m.fineByEngine[u]
		if f == nil {
			f = &fineEntry{engine: u, parent: parent, content: unitHash(e.TokenIDs[i*bs : (i+1)*bs]), copies: map[uint]int32{}}
			m.fineByEngine[u] = f
			m.fineByContent[f.content] = append(m.fineByContent[f.content], f)
		}
		f.copies[g]++
		f.groups |= bit
		m.stats.Stored++
		parent = u
	}
}

func (m *Mirror) removeFineLocked(u uint64, g uint) {
	f := m.fineByEngine[u]
	if f == nil {
		return
	}
	m.stats.Removed++
	if f.copies[g] > 1 {
		f.copies[g]--
		return
	}
	delete(f.copies, g)
	f.groups &^= 1 << g
	if f.groups != 0 {
		return
	}
	delete(m.fineByEngine, u)
	list := m.fineByContent[f.content]
	for i, x := range list {
		if x == f {
			list = append(list[:i], list[i+1:]...)
			break
		}
	}
	if len(list) == 0 {
		delete(m.fineByContent, f.content)
	} else {
		m.fineByContent[f.content] = list
	}
}

// Cached returns how many leading tokens of a prompt the engine would serve from its
// prefix cache now: the better of the block-level and the fine-grained tail hit.
func (m *Mirror) Cached(k PromptKeys) int {
	base := m.CachedBlockKeys(k.Blocks)
	m.mu.Lock()
	defer m.mu.Unlock()
	if k.Unit == 0 || len(m.fineByEngine) == 0 || m.attnMask == 0 {
		return base
	}
	full := 0 // contiguous full blocks every attention group holds
	for _, key := range k.Blocks {
		b := m.byKey[key]
		if b == nil || b.groups&m.attnMask != m.attnMask {
			break
		}
		full++
	}
	for j := len(k.Units) - 1; j >= 1; j-- {
		p := j * k.Unit
		if p <= base {
			break
		}
		if p/m.blockSize > full {
			continue
		}
		for _, ck := range m.fineByContent[k.Units[j-1]] {
			if m.mambaMask != 0 && ck.groups&m.mambaMask != m.mambaMask {
				continue
			}
			for _, at := range m.fineByContent[k.Units[j]] {
				if at.parent == ck.engine && at.groups&m.attnMask == m.attnMask {
					return p
				}
			}
		}
	}
	return base
}
