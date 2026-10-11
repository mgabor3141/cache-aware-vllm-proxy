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
// It reads the event format of vLLM 0.24.0 and later (events as maps tagged by "type").
//
// Engine block hashes are opaque (sha256 over pickled keys), so the mirror keys blocks by
// its own chain hash of token IDs: key(b) = H(key(parent), tokens(b)). An event's parent
// hash is mapped to a key through blocks the mirror has already seen; a block whose
// parent is unknown (stored before the proxy subscribed) is skipped, which only makes the
// mirror under-report hits. Blocks with extra_keys (images, LoRA, cache salt) hash with
// those keys too, so token IDs alone never match them. A prompt's keys are built at one
// block size and match only blocks stored at that size.
//
// Hybrid models (measured on Qwen3.8 Flash-Next, 2026-10-07): attention groups store a block as soon
// as it is prefilled; each recurrent (mamba) group keeps a state checkpoint at about one
// block per request. vLLM's hybrid coordinator (kv_cache_coordinator.py
// find_longest_cache_hit) takes the contiguous attention blocks, drops the last one when
// speculative decoding is on (the EAGLE drop; MTP flags every group), then
// the newest block at or below that where every mamba group holds a checkpoint.
//
// Unit hits. Engines that cache hybrid-model prefixes at a finer unit than the block
// (vLLM's --prefix-match-unit; halt95's vLLM fork serves Qwen3.8 Flash-Next with
// --block-size 4096 --prefix-match-unit 64) hash a prompt at every unit boundary, each
// hash chained over the whole prefix, and register where each prompt ends as unit
// entries. With speculative decoding that is an attention entry at the prompt's last unit
// boundary q and a recurrent checkpoint one unit lower, at p = q - unit; without it, both
// at the last unit boundary before the last token. Each is published as a BlockStored of
// block_size = unit carrying the last unit of tokens before it, its parent being the hash
// one unit lower. A later prompt that extends the prefix finds them: the engine
// (single_type_kv_cache_manager.py FullAttentionManager.find_longest_cache_hit) matches
// the contiguous full blocks, probes the unit boundaries inside the next block highest
// first, drops one unit (not a block) for speculative decoding, and the recurrent groups
// need a checkpoint there. A turn resuming at a previous turn's tail is so cached to, for
// example, 2 blocks and 31 units (10,176 tokens) rather than 2 blocks.
//
// The parents of unit entries are spans the engine never announces, so the mirror cannot
// chain them, and their tokens alone do not say where in a prompt they are (agent
// conversations repeat themselves). The mirror therefore anchors unit entries on prompts
// it has keyed: NewPromptKeys chains a key over every unit boundary of a prompt and
// remembers, under its keys at the two boundaries where it ends, the tokens of the entries
// the engine will register there. Cached then follows the engine. The attention groups
// reach through the contiguous cached full blocks, and further to the highest remembered
// end inside the next block whose attention entry the engine holds; the speculative drop
// takes one unit off that. The hit is the newest boundary at or below it where every
// recurrent group has a checkpoint: a cached block's end, or a remembered end whose
// checkpoint entry the engine holds, in any block. A turn whose predecessor's decode
// filled the block it ended in reaches past that block (the decoded block matches) and
// resumes at the checkpoint below it. Unit matching only adds boundaries, so a prediction
// is never below the block-level rule's. What is not verified is that the engine's entries
// are that earlier prompt's and not another's with the same unit of tokens; a prompt the
// mirror never keyed (sent around the proxy, or before it started) gets no unit hit.
package kvmirror

import (
	"container/list"
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

// copies is what the engine holds under one hash: a bit per KV cache group holding at
// least one copy, and the copies per group. vLLM can hold several physical blocks with one
// hash (a hybrid model recomputes attention blocks past its last recurrent checkpoint),
// and evicting one copy publishes BlockRemoved for the hash while another stays cached.
type copies struct {
	groups uint64
	n      map[uint]int32
}

func (c *copies) add(g uint, n int32) {
	if c.n == nil {
		c.n = map[uint]int32{}
	}
	c.n[g] += n
	c.groups |= 1 << g
}

// remove drops one copy of group g and reports whether any copy of any group is left.
func (c *copies) remove(g uint) bool {
	if c.n[g] > 1 {
		c.n[g]--
		return true
	}
	delete(c.n, g)
	c.groups &^= 1 << g
	return c.groups != 0
}

type mirrorBlock struct {
	key BlockKey
	copies
}

// unitEntry is one unit entry: where a prompt ended, in attention or recurrent groups.
type unitEntry struct {
	engine  uint64
	content uint64 // unitHash of its tokens
	copies
}

type Mirror struct {
	mu        sync.Mutex
	blockSize int                     // set by SetGeometry, else learned from the first stored block
	byEngine  map[uint64]*mirrorBlock // engine block hash -> block
	byKey     map[BlockKey]*mirrorBlock
	attnMask  uint64 // groups that are attention (need a contiguous prefix)
	mambaMask uint64 // groups that are recurrent state (need the hit block only)
	// unit is the engine's prefix-match unit from its metrics (SetGeometry); without it,
	// eventUnit is the size of the first unit-sized blocks the events showed.
	unit, eventUnit int
	units           map[uint64]*unitEntry            // engine hash -> unit entry
	unitsByContent  map[uint64]map[uint64]*unitEntry // unitHash -> engine hash -> unit entry
	// tails are where prompts the mirror keyed end (NewPromptKeys): facts about prompts,
	// not about the engine, so they outlive resets; the least recently used go first.
	tails     map[uint64]*list.Element // of *tail
	tailOrder *list.List
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
	// unsupported is a group kind the mirror cannot model (only full and MLA attention and
	// mamba are); while set, Cached returns 0.
	unsupported string
	sizeWarned  bool // a block of another size than blockSize was logged
	unitWarned  bool // an unusable prefix_match_unit was logged
	// dropLast applies the EAGLE drop (speculative decoding on): the last block, or the
	// last unit of a unit hit.
	dropLast bool
	stats    Stats
}

// PromptKeys are a prompt's mirror keys: one chained key per full block, at the block
// size they were built with, and with a prefix-match unit one chained key per unit
// boundary (Units[i] is the boundary (UnitsFrom+i+1)*Unit). Keys built at another block
// size than the mirror's match nothing, at another unit no unit hit.
type PromptKeys struct {
	BlockSize int
	Blocks    []BlockKey
	Unit      int
	UnitsFrom int
	Units     []uint64
}

func (k PromptKeys) Empty() bool { return len(k.Blocks) == 0 && len(k.Units) == 0 }

// unitKey returns the prompt's key at unit boundary x, if the keys hold it.
func (k PromptKeys) unitKey(x int) (uint64, bool) {
	i := x/k.Unit - 1 - k.UnitsFrom
	if i < 0 || i >= len(k.Units) {
		return 0, false
	}
	return k.Units[i], true
}

// Tail keeps only the unit keys from a unit below the prompt's last full block boundary
// on: where the engine registers this prompt's own end, the one unit hit it can have once
// it has run. Call it when the prompt is admitted, so a running request or an idle session
// keeps at most a block's worth of unit keys (66 at 4096/64) whatever the prompt's length.
func (k PromptKeys) Tail() PromptKeys {
	if k.Unit <= 0 {
		return k
	}
	from := len(k.Blocks)*k.BlockSize/k.Unit - 2
	if cut := from - k.UnitsFrom; cut > 0 {
		k.Units = append([]uint64(nil), k.Units[min(cut, len(k.Units)):]...)
		k.UnitsFrom = from
	}
	return k
}

func unitHash(tokens []int64) uint64 {
	h := sha256.New()
	var b [8]byte
	for _, t := range tokens {
		binary.LittleEndian.PutUint64(b[:], uint64(t))
		h.Write(b[:])
	}
	return binary.LittleEndian.Uint64(h.Sum(nil))
}

// end is an entry the engine will register where a prompt ends, keyed by the prompt's
// unit key at its boundary: the token hash of the unit before the boundary (what the
// entry's BlockStored carries), and whether it is the attention entry, the checkpoint,
// or both (without the speculative drop they share the boundary).
type end struct {
	content    uint64
	attn, ckpt bool
}

type tail struct {
	key uint64
	end
}

// maxTails bounds the prompt ends remembered: far more than an engine's cache holds.
// The least recently remembered or looked up go first.
const maxTails = 1 << 15

// NewPromptKeys computes a prompt's keys at the mirror's block size (none while it is
// unknown) and prefix-match unit, and remembers where the engine will register the
// prompt's end, so a later prompt extending it can be matched there. Unit keys run up to
// the last boundary a hit can use: the prompt's last full unit with the speculative drop,
// else the last before its last token.
func (m *Mirror) NewPromptKeys(prompt []int64) PromptKeys {
	m.mu.Lock()
	bs, unit, drop := m.blockSize, m.unitLocked(), m.dropLast
	m.mu.Unlock()
	k := PromptKeys{BlockSize: bs, Blocks: PromptBlockKeys(prompt, bs)}
	if unit <= 0 || bs <= 0 {
		return k
	}
	limit := len(prompt)
	if !drop {
		limit--
	}
	k.Unit = unit
	key := rootKey
	for i := 0; (i+1)*unit <= limit; i++ {
		key = chainKey(key, prompt[i*unit:(i+1)*unit], nil)
		k.Units = append(k.Units, binary.LittleEndian.Uint64(key[:8]))
	}
	q := limit / unit * unit // the attention entry; with the drop the checkpoint is a unit lower
	p := q
	if drop {
		p -= unit
	}
	if p >= unit {
		m.mu.Lock()
		m.rememberLocked(k.Units[q/unit-1], unitHash(prompt[q-unit:q]), true, false)
		m.rememberLocked(k.Units[p/unit-1], unitHash(prompt[p-unit:p]), false, true)
		m.mu.Unlock()
	}
	return k
}

func (m *Mirror) rememberLocked(key, content uint64, attn, ckpt bool) {
	if el, ok := m.tails[key]; ok {
		t := el.Value.(*tail)
		t.attn, t.ckpt = t.attn || attn, t.ckpt || ckpt
		m.tailOrder.MoveToBack(el)
		return
	}
	if m.tailOrder.Len() == maxTails {
		delete(m.tails, m.tailOrder.Remove(m.tailOrder.Front()).(*tail).key)
	}
	m.tails[key] = m.tailOrder.PushBack(&tail{key: key, end: end{content: content, attn: attn, ckpt: ckpt}})
}

// SetSpeculative tells the mirror the engine runs speculative decoding (EAGLE, MTP), so
// a hit loses its last block, or its last unit with a prefix-match unit, as the engine's
// does. Off by default: plain vLLM serves every cached block.
func (m *Mirror) SetSpeculative(on bool) {
	m.mu.Lock()
	m.dropLast = on
	m.mu.Unlock()
}

// SetGeometry sets the engine's block size and prefix-match unit (from its metrics), so
// prompt keys are built right before the first event shows them. A unit that is not a
// proper divisor of the block size is ignored (and logged once); 0 leaves the unit to
// the events.
func (m *Mirror) SetGeometry(blockSize, unit int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if blockSize <= 0 {
		return
	}
	m.blockSize = blockSize
	if unit > 0 && (unit >= blockSize || blockSize%unit != 0) {
		if unit != blockSize && !m.unitWarned {
			m.unitWarned = true
			log.Printf("level=warn msg=\"prefix_match_unit is not a divisor of the block size: no unit hits\" prefix_match_unit=%d block_size=%d", unit, blockSize)
		}
		unit = 0
	}
	m.unit = unit
}

// unitLocked is the engine's prefix-match unit: from its metrics, else from its events;
// 0 when there is none.
func (m *Mirror) unitLocked() int {
	if m.unit > 0 {
		return m.unit
	}
	return m.eventUnit
}

// isUnitLocked reports whether blocks of size are unit entries. Without a unit from the
// metrics, the first size that divides the block size is taken for the unit (until the
// mirror resets).
func (m *Mirror) isUnitLocked(size int) bool {
	if m.unitLocked() == 0 && size < m.blockSize && m.blockSize%size == 0 {
		m.eventUnit = size
		log.Printf("level=info msg=\"kv events show a prefix-match unit\" unit=%d block_size=%d", size, m.blockSize)
	}
	return size == m.unitLocked()
}

type Stats struct {
	// Reports are reuse reports (kv_cache_report_mode "full"), not counted as copies.
	Stored, Removed, Cleared, Restarts, SkippedOrphan, SkippedEmpty, Batches, Duplicates, SeqGaps, Reports uint64
}

func New() *Mirror {
	m := &Mirror{tails: map[uint64]*list.Element{}, tailOrder: list.New()}
	m.resetLocked()
	return m
}

// errResync ends a subscription so it restarts with a replay (a gap or an engine restart).
var errResync = fmt.Errorf("kv events sequence gap or restart: resyncing")

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
// backwards means the engine restarted, so the mirror starts over. It returns false, and
// the mirror is unhealthy, on a gap (batches were lost) or a restart: either way the
// mirror can no longer be trusted until it resyncs from a replay. A restart's batch is
// not applied; the replay brings it.
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
			log.Printf("level=warn msg=\"kv events sequence went back: engine restarted, mirror cleared, resyncing\" seq=%d last=%d", seq, m.lastSeq)
			m.resetLocked()
			m.stats.Restarts++
			m.connected = false
			return false
		case seq > m.lastSeq+1:
			m.stats.SeqGaps++
			m.lastSeq = seq
			m.connected = false
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

// resetLocked forgets every block and unit entry, what the groups are and the unit the
// events showed (a restarted engine may serve another model); the geometry set from
// metrics stays.
func (m *Mirror) resetLocked() {
	m.clearLocked()
	m.attnMask, m.mambaMask = 0, 0
	m.unsupported = ""
	m.eventUnit = 0
	m.firstSeq, m.lastSeq, m.replayedTo, m.liveSeq = -1, -1, -1, -1
}

// clearLocked forgets every block and unit entry.
func (m *Mirror) clearLocked() {
	m.generation++
	m.byEngine = map[uint64]*mirrorBlock{}
	m.byKey = map[BlockKey]*mirrorBlock{}
	m.units = map[uint64]*unitEntry{}
	m.unitsByContent = map[uint64]map[uint64]*unitEntry{}
}

func (m *Mirror) applyLocked(e *Event) {
	if e.Medium != nil && *e.Medium != "" && *e.Medium != "GPU" {
		return
	}
	switch e.Type {
	case "AllBlocksCleared":
		m.clearLocked()
		m.stats.Cleared++
	case "BlockRemoved":
		if e.GroupIdx == nil {
			return
		}
		g := uint(*e.GroupIdx)
		for _, h := range e.BlockHashes {
			u, ok := hashToUint(h)
			if !ok {
				continue
			}
			if b := m.byEngine[u]; b != nil {
				m.stats.Removed++
				if !b.remove(g) {
					delete(m.byEngine, u)
					if m.byKey[b.key] == b {
						delete(m.byKey, b.key)
					}
				}
			} else if f := m.units[u]; f != nil {
				m.stats.Removed++
				if !f.remove(g) {
					m.dropUnitLocked(f)
				}
			}
		}
	case "BlockStored":
		if e.GroupIdx == nil || e.BlockSize <= 0 {
			return
		}
		g := uint(*e.GroupIdx)
		bit := uint64(1) << g
		switch e.Kind {
		case "mamba":
			m.mambaMask |= bit
		case "full_attention", "mla_attention", "":
			m.attnMask |= bit
		default:
			// sliding_window, chunked_local_attention, cross_attention and the like hit
			// with gaps (vLLM single_type_kv_cache_manager.py); the mirror does not model
			// them, so it predicts nothing cached rather than something wrong.
			if m.unsupported == "" {
				log.Printf("level=error msg=\"kv cache group kind not supported: the mirror predicts nothing cached\" kind=%q group=%d", e.Kind, g)
			}
			m.unsupported = e.Kind
			return
		}
		if m.blockSize == 0 {
			m.blockSize = e.BlockSize
		}
		if len(e.BlockHashes) == 0 {
			m.stats.SkippedEmpty++
			return
		}
		if e.ParentBlockHash == nil && e.Kind == "mamba" && len(e.BlockHashes) > 1 {
			// A reuse report for a recurrent group lists every block from 0 (null
			// placeholders included) though state is held only at the checkpoint; a real
			// recurrent store is one block with its parent.
			m.stats.Reports++
			return
		}
		if e.BlockSize != m.blockSize {
			if m.isUnitLocked(e.BlockSize) {
				m.storeUnitsLocked(e, g)
				return
			}
			if !m.sizeWarned {
				// Keys are built at the engine's block size; blocks of another size never match.
				m.sizeWarned = true
				log.Printf("level=warn msg=\"kv events block size differs from the engine's and its unit: those blocks never match\" event_block_size=%d block_size=%d unit=%d group=%d", e.BlockSize, m.blockSize, m.unitLocked(), g)
			}
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
			if e.ParentBlockHash == nil && m.hashHeldLocked(u, g) {
				// A reuse report (kv_cache_report_mode "full", block_pool.py
				// emit_cached_block_events) lists the blocks a request hit, from block 0,
				// with no BlockRemoved ever to match: not new copies. A real second copy
				// of a block stored from 0 is missed too, which only under-reports.
				m.stats.Reports++
				continue
			}
			b := m.byKey[key]
			if b == nil {
				b = &mirrorBlock{key: key}
				m.byKey[key] = b
			}
			m.byEngine[u] = b
			b.add(g, 1)
			m.stats.Stored++
			if f := m.units[u]; f != nil {
				// A unit entry at a block's end (a checkpoint there) has the block's hash:
				// its copies are the block's.
				for fg, n := range f.n {
					b.add(fg, n)
				}
				m.dropUnitLocked(f)
			}
		}
	}
}

// storeUnitsLocked records a unit-sized BlockStored. An entry whose hash is a known
// block's (a checkpoint at a block's end) is a copy of that block; entries with extra
// keys are skipped, as tokens alone never match them.
func (m *Mirror) storeUnitsLocked(e *Event, g uint) {
	size := e.BlockSize
	for i, h := range e.BlockHashes {
		if (i+1)*size > len(e.TokenIDs) {
			return
		}
		u, ok := hashToUint(h)
		switch {
		case !ok || i < len(e.ExtraKeys) && e.ExtraKeys[i] != nil:
		case e.ParentBlockHash == nil && m.hashHeldLocked(u, g):
			m.stats.Reports++ // a reuse report, as in applyLocked
		case m.byEngine[u] != nil:
			m.byEngine[u].add(g, 1)
			m.stats.Stored++
		default:
			f := m.units[u]
			if f == nil {
				f = &unitEntry{engine: u, content: unitHash(e.TokenIDs[i*size : (i+1)*size])}
				m.units[u] = f
				if m.unitsByContent[f.content] == nil {
					m.unitsByContent[f.content] = map[uint64]*unitEntry{}
				}
				m.unitsByContent[f.content][u] = f
			}
			f.add(g, 1)
			m.stats.Stored++
		}
	}
}

// hashHeldLocked reports whether the engine hash is a block or unit entry held in group g.
func (m *Mirror) hashHeldLocked(u uint64, g uint) bool {
	if b := m.byEngine[u]; b != nil {
		return b.groups&(1<<g) != 0
	}
	f := m.units[u]
	return f != nil && f.groups&(1<<g) != 0
}

func (m *Mirror) dropUnitLocked(f *unitEntry) {
	delete(m.units, f.engine)
	if c := m.unitsByContent[f.content]; c != nil {
		delete(c, f.engine)
		if len(c) == 0 {
			delete(m.unitsByContent, f.content)
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
// prefix cache now. As the engine does, it finds how far the attention groups reach (the
// contiguous full blocks, with a prefix-match unit then the unit boundaries of the next
// block, highest first), drops one block (one unit with a unit) for speculative decoding,
// and returns the newest boundary at or below that where every recurrent group holds a
// checkpoint: a cached block's end, or with a unit a remembered prompt end whose
// checkpoint the engine holds, in any block below the reach. Unit matching only adds
// boundaries to the block-level rule, so it never predicts less.
func (m *Mirror) Cached(k PromptKeys) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if k.BlockSize <= 0 || k.BlockSize != m.blockSize || m.attnMask == 0 || m.unsupported != "" {
		return 0
	}
	bs := k.BlockSize
	c := 0
	for _, key := range k.Blocks {
		b := m.byKey[key]
		if b == nil || b.groups&m.attnMask != m.attnMask {
			break
		}
		c++
	}
	step := bs
	if k.Unit > 0 && k.Unit == m.unitLocked() && m.mambaMask != 0 {
		step = k.Unit // vLLM serves unit hits only to hybrid models (aligned recurrent state)
	}
	reach := c * bs
	if step < bs {
		reach = max(reach, m.attentionReachLocked(k, c))
	}
	if m.dropLast && reach > 0 {
		reach -= step
	}
	for x := reach / step * step; x > 0; x -= step {
		if m.checkpointLocked(k, x) {
			return x
		}
	}
	return 0
}

// attentionReachLocked is the highest remembered prompt end inside block c (past the c
// cached full blocks) whose attention entry the engine holds, or 0.
func (m *Mirror) attentionReachLocked(k PromptKeys, c int) int {
	u, bs := k.Unit, k.BlockSize
	for q := (c+1)*bs - u; q > c*bs; q -= u {
		if e, ok := m.endLocked(k, q); ok && e.attn && m.heldLocked(e.content, m.attnMask) {
			return q
		}
	}
	return 0
}

// checkpointLocked reports whether every recurrent group holds a checkpoint at the
// prompt's boundary x (within its cached prefix): a cached block ending there, or a
// remembered prompt end whose checkpoint entry the engine holds. A checkpoint at a block's
// end has the block's hash and so is the block's.
func (m *Mirror) checkpointLocked(k PromptKeys, x int) bool {
	if m.mambaMask == 0 {
		return true
	}
	if x%k.BlockSize == 0 {
		b := m.byKey[k.Blocks[x/k.BlockSize-1]]
		return b != nil && b.groups&m.mambaMask == m.mambaMask
	}
	e, ok := m.endLocked(k, x)
	return ok && e.ckpt && m.heldLocked(e.content, m.mambaMask)
}

// endLocked is the remembered prompt end at the prompt's unit boundary x, if any.
func (m *Mirror) endLocked(k PromptKeys, x int) (end, bool) {
	key, ok := k.unitKey(x)
	if !ok {
		return end{}, false
	}
	el, ok := m.tails[key]
	if !ok {
		return end{}, false
	}
	m.tailOrder.MoveToBack(el) // looked up: a conversation that comes back keeps its ends
	return el.Value.(*tail).end, true
}

// heldLocked reports whether a unit entry with these tokens is held in every group of mask.
func (m *Mirror) heldLocked(content, mask uint64) bool {
	for _, f := range m.unitsByContent[content] {
		if f.groups&mask == mask {
			return true
		}
	}
	return false
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
	Unit          int     `json:"unit"`
	UnitEntries   int     `json:"unit_entries"`
	Unsupported   string  `json:"unsupported_kind,omitempty"` // a group kind the mirror cannot model: nothing is predicted cached
	LastEventAgeS float64 `json:"last_event_age_s"`
	Stats         Stats   `json:"stats"`
}

func (m *Mirror) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	s := Status{Connected: m.connected, Blocks: len(m.byKey), BlockSize: m.blockSize, Unit: m.unitLocked(), UnitEntries: len(m.units), Unsupported: m.unsupported, Stats: m.stats}
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
			return errResync
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
		// Skipped, but its sequence is consumed: a resync would meet it again.
		log.Printf("level=warn msg=\"kv events batch undecodable, skipped (vLLM older than 0.24?)\" seq=%d err=%v", seq, err)
		return seq, nil, seq >= 0
	}
	return seq, events, true
}

// skipHole accepts that the batches from up to (not including) to are lost for good and
// forgets every block: what they removed is unknown, and only forgetting under-reports.
func (m *Mirror) skipHole(from, to int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	log.Printf("level=warn msg=\"kv events replay buffer has a hole, mirror cleared\" from=%d to=%d", from, to)
	m.stats.SeqGaps++
	m.clearLocked()
	if m.lastSeq >= 0 {
		m.lastSeq = to - 1
	}
}

// replay rebuilds recent state from the publisher's replay buffer (its last
// buffer_steps batches). The publisher's ROUTER queues at most ~1,000 messages for a
// reader and drops the rest silently, so a replay is read in rounds: each takes the
// batches that arrive in sequence, stops at the first gap or a quiet second, and asks
// again from the next missing sequence, until the end-of-sequence marker (seq -1). A round
// whose first batch is past the sequence asked for skips the hole and clears the mirror:
// those batches are gone, and may have removed blocks.
// Undecodable batches are skipped too. A replay that does not reach the marker is an error.
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
			if seq < next {
				continue
			}
			if seq > next && n > 0 {
				return n, next, false, nil // the publisher dropped batches: ask again from here
			}
			if seq > next && next != 0 {
				// Asked from next, the buffer's first batch is later: those batches are
				// gone for good (dropped at the publisher's high-water mark). They may have
				// removed blocks, so the mirror starts over from the batch after them.
				m.skipHole(next, seq)
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
