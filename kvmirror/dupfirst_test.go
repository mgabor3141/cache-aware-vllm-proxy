package kvmirror

import "testing"

// Shape seen live on the hybrid MTP engine (halt95 fork v2.5.1, attention group 4, mamba
// groups 0-2): a prompt sharing block 0 with a cached one, but with no recurrent checkpoint
// to resume from, is prefilled from 0, and the engine stores a second physical copy of
// block 0: a BlockStored with no parent whose hash the mirror already holds. That is a real
// copy. Taking it for a reuse report and not counting it made the next eviction of one
// copy drop block 0 from the mirror while the engine still held it, so every prompt
// sharing it was predicted cold (production regression, 2026-10-11).
func TestRegressionSecondCopyOfFirstBlock(t *testing.T) {
	bs := 4
	p := seqPrompt(2*bs + 1)
	m := New()
	evs := []Event{stored(4, "mla_attention", bs, 1, nil, p[:bs]), stored(4, "mla_attention", bs, 2, uint64(1), p[bs:2*bs])}
	for g := 0; g < 3; g++ {
		evs = append(evs, stored(g, "mamba", bs, 2, uint64(1), p[bs:2*bs]))
	}
	m.Apply(0, evs)
	if got := m.CachedTokens(p); got != 2*bs {
		t.Fatalf("cached %d, want %d", got, 2*bs)
	}
	// A recompute of block 0: a second copy, then one copy evicted.
	m.Apply(1, []Event{stored(4, "mla_attention", bs, 1, nil, p[:bs])})
	m.Apply(2, []Event{removed(4, 1)})
	if got := m.CachedTokens(p); got != 2*bs {
		t.Fatalf("one of two copies of block 0 evicted: cached %d, want %d", got, 2*bs)
	}
}
