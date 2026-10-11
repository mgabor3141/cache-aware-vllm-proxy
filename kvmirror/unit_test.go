package kvmirror

import (
	"bytes"
	"log"
	"os"
	"strings"
	"testing"
)

// The deployed geometry: halt95's vLLM fork with --block-size 4096 --prefix-match-unit 64,
// Qwen3.8 Flash-Next's groups as in testdata/kvevents (0-3 recurrent, 5 attention).
const (
	ubs  = 4096
	unit = 64
)

var mambaGroups = []int{0, 1, 2, 3}

func stored(g int, kind string, size int, hash uint64, parent any, toks []int64) Event {
	return Event{Type: "BlockStored", BlockHashes: []any{hash}, ParentBlockHash: parent, TokenIDs: toks, BlockSize: size, GroupIdx: &g, Kind: kind}
}

func removed(g int, hash uint64) Event {
	return Event{Type: "BlockRemoved", BlockHashes: []any{hash}, GroupIdx: &g}
}

// fullBlocks announces a prompt's first n blocks in the attention group, hashes 1..n.
func fullBlocks(toks []int64, n int) []Event {
	var evs []Event
	var parent any
	for i := 0; i < n; i++ {
		evs = append(evs, stored(5, "mla_attention", ubs, uint64(i+1), parent, toks[i*ubs:(i+1)*ubs]))
		parent = uint64(i + 1)
	}
	return evs
}

// tailEntries is what the engine registers where a prompt of n tokens ended, with
// speculative decoding (vLLM _cache_partial_tail_block and
// get_mamba_prefill_checkpoint_position): an attention entry at the last unit boundary q
// and a checkpoint in every recurrent group one unit lower, each a unit-sized BlockStored
// carrying the unit before it, its parent the hash one unit lower. ck and at are their
// engine hashes (a checkpoint at a block boundary has the block's); ckParent is the
// checkpoint's parent, a hash the engine never announces. A prompt ending on a block
// boundary leaves no attention entry: the block is full.
func tailEntries(toks []int64, n int, ck, at uint64, ckParent any) []Event {
	q := n / unit * unit
	p := q - unit
	var evs []Event
	for _, g := range mambaGroups {
		evs = append(evs, stored(g, "mamba", unit, ck, ckParent, toks[p-unit:p]))
	}
	if q%ubs != 0 {
		evs = append(evs, stored(5, "mla_attention", unit, at, ck, toks[p:q]))
	}
	return evs
}

func fixtureTokens(t *testing.T) []int64 {
	t.Helper()
	_, reqs := loadFixture(t)
	return reqs[0].Tokens // a real 19,557-token conversation
}

func quietLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	return &logs
}

// The shape the deployed engine reports: a turn whose prompt ended at 10,260 tokens left
// an attention entry at 10,240 and a checkpoint at 10,176; the next turn (10.8K tokens,
// extending it) is cached to 10,176 = 2 blocks + 31 units, and so is a resend. Unit-sized
// events are the engine's unit, not a block size mismatch.
func TestKVMirrorUnitHitDeployedShape(t *testing.T) {
	logs := quietLog(t)
	toks := fixtureTokens(t)
	turn1, turn2 := toks[:10260], toks[:10800]
	m := New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(turn1) // turn 1 went through the proxy
	m.Apply(0, fullBlocks(turn1, 2))
	m.Apply(1, tailEntries(turn1, len(turn1), 900, 901, uint64(899)))
	for _, p := range [][]int64{turn2, turn1} {
		if got := m.CachedTokens(p); got != 2*ubs+31*unit {
			t.Fatalf("%d-token prompt: cached %d, want 10176", len(p), got)
		}
	}
	if strings.Contains(logs.String(), "differs") {
		t.Fatalf("unit events logged as a block size mismatch:\n%s", logs)
	}
	if st := m.Status(); st.Unit != unit || st.UnitEntries != 2 {
		t.Fatalf("status %+v", st)
	}
	// A prompt that diverges before the end does not hit, although the fixture repeats
	// itself and holds the end's 128 tokens at 9,152 too (block 2, where the engine probes).
	other := append([]int64(nil), turn2...)
	other[10200]++
	if got := m.CachedTokens(other); got != 0 {
		t.Fatalf("diverging prompt: cached %d", got)
	}
	// The checkpoint is needed in every recurrent group: losing one leaves the full
	// blocks, which have no checkpoint.
	m.Apply(2, []Event{removed(2, 900)})
	if got := m.CachedTokens(turn2); got != 0 {
		t.Fatalf("checkpoint evicted in one group: cached %d", got)
	}
	m.Apply(3, []Event{stored(2, "mamba", unit, 900, uint64(899), turn1[10112:10176])})
	if got := m.CachedTokens(turn2); got != 10176 {
		t.Fatalf("checkpoint stored again: cached %d", got)
	}
	// A unit hit extends cached full blocks: without block 1 it is gone.
	m.Apply(4, []Event{removed(5, 2)})
	if got := m.CachedTokens(turn2); got != 0 {
		t.Fatalf("block 1 evicted: cached %d", got)
	}
}

// Without prefix_match_unit in the metrics, the unit is the size of the unit-sized
// events; it is forgotten when the mirror resets (an engine restart may change it).
func TestKVMirrorUnitFromEvents(t *testing.T) {
	quietLog(t)
	toks := fixtureTokens(t)
	turn1, turn2 := toks[:10260], toks[:10800]
	m := New()
	m.SetGeometry(ubs, 0)
	m.Apply(0, fullBlocks(turn1, 2))
	m.Apply(1, tailEntries(turn1, len(turn1), 900, 901, uint64(899)))
	m.NewPromptKeys(turn1) // keyed once the unit is known
	if got := m.CachedTokens(turn2); got != 10176 {
		t.Fatalf("unit learned from events: cached %d, want 10176", got)
	}
	m.Apply(0, nil) // the engine restarted
	if st := m.Status(); st.Unit != 0 || st.UnitEntries != 0 {
		t.Fatalf("after a restart: %+v", st)
	}
	// The metrics' unit wins over other small sizes; a unit that does not divide the
	// block size is refused.
	logs := quietLog(t)
	m.SetGeometry(ubs, unit)
	m.Apply(1, []Event{stored(0, "mamba", 16, 77, nil, toks[:16])})
	if st := m.Status(); st.Unit != unit || st.UnitEntries != 0 || !strings.Contains(logs.String(), "differs") {
		t.Fatalf("16-token event with unit 64 from metrics: %+v\n%s", st, logs)
	}
	m.SetGeometry(ubs, 100)
	if st := m.Status(); st.Unit != 0 || !strings.Contains(logs.String(), "not a divisor") {
		t.Fatalf("unit 100 with block 4096: %+v\n%s", st, logs)
	}
}

// Unit entries are forgotten with the blocks: on AllBlocksCleared, a restart, and a
// resubscription (resetLocked).
func TestKVMirrorUnitEntriesReset(t *testing.T) {
	toks := fixtureTokens(t)
	for _, reset := range []func(m *Mirror){
		func(m *Mirror) { m.Apply(2, []Event{{Type: "AllBlocksCleared"}}) },
		func(m *Mirror) { m.Apply(0, nil) },
		func(m *Mirror) { m.mu.Lock(); m.resetLocked(); m.mu.Unlock() },
	} {
		quietLog(t)
		m := New()
		m.SetGeometry(ubs, unit)
		m.Apply(0, fullBlocks(toks, 2))
		m.Apply(1, tailEntries(toks, 10260, 900, 901, uint64(899)))
		reset(m)
		if st := m.Status(); st.Blocks != 0 || st.UnitEntries != 0 || len(m.unitsByContent) != 0 {
			t.Fatalf("after a reset: %+v", st)
		}
	}
}

// A unit hit extends the prompt's cached full blocks and lands where a prompt the mirror
// keyed ended (review cv-52 #3): engine entries whose tokens recur in a prompt with
// nothing cached are no hit, nor are the ends of prompts the mirror never keyed. Inside
// the first block no full block is needed, as for the engine.
func TestKVMirrorUnitHitNeedsCachedBlocksAndAKnownEnd(t *testing.T) {
	toks := fixtureTokens(t)
	m := New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:10260])
	m.Apply(0, tailEntries(toks, 10260, 900, 901, uint64(899)))
	if got := m.CachedTokens(toks[:10800]); got != 0 {
		t.Fatalf("nothing cached: %d", got)
	}
	m.Apply(1, fullBlocks(toks, 1))
	if got := m.CachedTokens(toks[:10800]); got != 0 {
		t.Fatalf("an end in block 2 with only block 0 cached: %d", got)
	}
	m.Apply(2, tailEntries(toks, 2000, 800, 801, uint64(799)))
	if got := m.CachedTokens(toks[:2100]); got != 0 {
		t.Fatalf("the end of a prompt the mirror never keyed: %d", got)
	}
	m.NewPromptKeys(toks[:2000])
	if got := m.CachedTokens(toks[:2100]); got != 1920 {
		t.Fatalf("an end in block 0: cached %d, want 1920", got)
	}
}

// A turn that ended on a block boundary leaves no attention entry (the block is full)
// and its checkpoint one unit below the block's end; one that ended a unit past a block
// boundary has its checkpoint at the boundary, under the block's own hash (fork patch
// 0094). The engine drops one unit from the attention hit and finds those checkpoints.
func TestKVMirrorUnitHitAtBlockBoundaries(t *testing.T) {
	toks := fixtureTokens(t)
	m := New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:2*ubs+10])
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, tailEntries(toks, 2*ubs+10, 700, 701, uint64(699)))
	if got := m.CachedTokens(toks[:9000]); got != 2*ubs-unit {
		t.Fatalf("checkpoint below the block end: cached %d, want %d", got, 2*ubs-unit)
	}

	m = New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:2*ubs+unit+10])
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, tailEntries(toks, 2*ubs+unit+10, 2, 801, uint64(799)))
	if st := m.Status(); st.UnitEntries != 1 {
		t.Fatalf("the checkpoint at the block's end is the block's: %+v", st)
	}
	if got := m.CachedTokens(toks[:9000]); got != 2*ubs {
		t.Fatalf("checkpoint at the block boundary: cached %d, want %d", got, 2*ubs)
	}
	m.Apply(2, []Event{removed(0, 2)})
	if got := m.CachedTokens(toks[:9000]); got != 0 {
		t.Fatalf("its checkpoint evicted: cached %d", got)
	}
}

// Without the speculative drop, the engine registers the attention entry and the
// checkpoint at the same boundary (the last before the prompt's last token), so one
// entry held by every group is the hit.
func TestKVMirrorUnitHitWithoutDrop(t *testing.T) {
	toks := fixtureTokens(t)
	m := New()
	m.dropLast = false
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:10260])
	m.Apply(0, fullBlocks(toks, 2))
	p := (10260 - 1) / unit * unit
	var evs []Event
	for _, g := range append([]int{5}, mambaGroups...) {
		kind := "mamba"
		if g == 5 {
			kind = "mla_attention"
		}
		evs = append(evs, stored(g, kind, unit, 900, uint64(899), toks[p-unit:p]))
	}
	m.Apply(1, evs[1:])
	if got := m.CachedTokens(toks[:10800]); got != 0 {
		t.Fatalf("checkpoint without the attention entry: cached %d", got)
	}
	m.Apply(2, evs[:1])
	if got := m.CachedTokens(toks[:10800]); got != p {
		t.Fatalf("cached %d, want %d", got, p)
	}
}

// Many unit entries sharing the probed tokens (a low-entropy span cached under many
// prefixes) cost one lookup each at a remembered end, never one per pair (review cv-52
// #12), and an entry held in every group still hits.
func TestKVMirrorUnitHitAmongDuplicates(t *testing.T) {
	toks := append([]int64(nil), fixtureTokens(t)[:10800]...)
	for i := 10112; i < 10240; i++ {
		toks[i] = 0
	}
	m := New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:10260])
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, duplicates(toks, 2000))
	keys := m.NewPromptKeys(toks)
	if got := m.Cached(keys); got != 0 {
		t.Fatalf("no entry pairs up: cached %d", got)
	}
	m.Apply(2, tailEntries(toks, 10260, 900, 901, uint64(899)))
	if got := m.Cached(keys); got != 10176 {
		t.Fatalf("cached %d, want 10176", got)
	}
}

// duplicates are n attention entries holding the tokens of a 10,260-token prompt's end and
// n entries holding its checkpoint's tokens in only one of the four recurrent groups.
func duplicates(toks []int64, n uint64) []Event {
	var evs []Event
	for _, g := range mambaGroups[1:] {
		evs = append(evs, stored(g, "mamba", unit, 9000, nil, toks[:unit]))
	}
	for i := uint64(0); i < n; i++ {
		evs = append(evs, stored(0, "mamba", unit, 10000+i, uint64(5000), toks[10112:10176]))
		evs = append(evs, stored(5, "mla_attention", unit, 20000+i, uint64(7000), toks[10176:10240]))
	}
	return evs
}

// A prompt's unit keys cover what a waiter can hit; once admitted, Tail keeps only its
// last block's, where its own end is registered: bounded whatever the prompt's length
// (review cv-52 #13), and still enough to see its own end. The ends remembered are
// bounded too.
func TestPromptKeysTail(t *testing.T) {
	m := New()
	m.SetGeometry(ubs, unit)
	k := m.NewPromptKeys(seqPrompt(200000))
	if len(k.Units) != 200000/unit {
		t.Fatalf("waiting keys hold %d unit keys", len(k.Units))
	}
	if tail := k.Tail(); len(tail.Units) > ubs/unit+2 || len(tail.Blocks) != len(k.Blocks) {
		t.Fatalf("tail keeps %d unit keys, %d blocks", len(tail.Units), len(tail.Blocks))
	}

	toks := fixtureTokens(t)[:10260]
	tk := m.NewPromptKeys(toks).Tail()
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, tailEntries(toks, len(toks), 900, 901, uint64(899)))
	if got := m.Cached(tk); got != 10176 {
		t.Fatalf("tail keys: cached %d, want 10176", got)
	}

	for i := uint64(0); i < maxTails+10; i++ {
		m.rememberLocked(i, 0, true, false)
	}
	if _, oldest := m.tails[0]; len(m.tails) != maxTails || oldest {
		t.Fatalf("%d ends remembered", len(m.tails))
	}
}

// The captured events (all at the block size) predict the same with a unit configured.
func TestKVMirrorCapturedEventsWithUnit(t *testing.T) {
	batches, reqs := loadFixture(t)
	m := New()
	m.SetGeometry(3200, 64)
	for _, b := range batches {
		events, err := decodeBatch(b[8:])
		if err != nil {
			t.Fatal(err)
		}
		m.Apply(-1, events)
	}
	for i := 1; i <= 2; i++ {
		if got := m.CachedTokens(reqs[i].Tokens); got != reqs[i].Cached {
			t.Fatalf("request %d: mirror %d, engine %d", i, got, reqs[i].Cached)
		}
	}
}

func BenchmarkCachedUnitDuplicates(b *testing.B) {
	toks := seqPrompt(10800)
	for i := 10112; i < 10240; i++ {
		toks[i] = 0
	}
	m := New()
	m.SetGeometry(ubs, unit)
	m.NewPromptKeys(toks[:10260])
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, duplicates(toks, 800))
	keys := m.NewPromptKeys(toks)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Cached(keys)
	}
}

// decodeAcross is what the engine publishes when a request's decode fills the block its
// prompt ended in: the block becomes a full block hashed over prompt and output (hash
// next, parent the previous block's), and with promote the unit entry it held is removed
// (vLLM block_pool cache_full_blocks: partial-to-full promotion).
func decodeAcross(all []int64, block int, next uint64, promote bool, at uint64) []Event {
	var evs []Event
	if promote {
		evs = append(evs, removed(5, at))
	}
	return append(evs, stored(5, "mla_attention", ubs, next, uint64(block), all[block*ubs:(block+1)*ubs]))
}

// Production, block 4096, unit 64: a 64,725-token prompt ends with its checkpoint at
// 64,640 (attention entry at 64,704), its 2,309-token decode crosses 65,536 and fills
// block 15, and the next turn (67,710 tokens, the previous prompt and output verbatim)
// is cached by the engine to 64,640: its attention reaches through block 15, and the
// recurrent groups resume at the checkpoint, a block below the probed one. The mirror
// predicted 0. Whether or not the engine drops the unit entry when the block fills, and
// whether or not the next turn's block 15 matches the decoded one, the prediction is the
// checkpoint, and never below the block-level rule.
func TestKVMirrorUnitHitAfterDecodeFillsTheBlock(t *testing.T) {
	all := seqPrompt(67710)
	prev, next := all[:64725], all
	for _, c := range []struct {
		name             string
		promote, rewrite bool
		want             int
	}{
		{"block filled, entry kept", false, false, 64640},
		{"block filled, entry removed", true, false, 64640},
		{"output re-rendered, entry kept", false, true, 64640},
		{"output re-rendered, entry removed", true, true, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			m := New()
			m.SetGeometry(ubs, unit)
			m.NewPromptKeys(prev)
			m.Apply(0, fullBlocks(all, 15))
			m.Apply(1, tailEntries(prev, len(prev), 900, 901, uint64(899)))
			m.Apply(2, decodeAcross(all, 15, 16, c.promote, 901))
			p := append([]int64(nil), next...)
			if c.rewrite {
				p[64800]++ // the next turn renders the previous output differently
			}
			keys := m.NewPromptKeys(p)
			if got := m.Cached(keys); got != c.want {
				t.Fatalf("cached %d, want %d", got, c.want)
			}
			blocks := keys
			blocks.Unit, blocks.Units = 0, nil
			for _, g := range mambaGroups { // a checkpoint at block 15's start
				m.Apply(int64(3+g), []Event{stored(g, "mamba", ubs, 15, uint64(14), all[14*ubs:15*ubs])})
			}
			if got, floor := m.Cached(keys), m.Cached(blocks); got < floor || !c.rewrite && floor != 15*ubs {
				t.Fatalf("cached %d, below the block-level %d", got, floor)
			}
		})
	}
}
