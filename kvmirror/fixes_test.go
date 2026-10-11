package kvmirror

import (
	"os"
	"strings"
	"testing"
)

func fixPrompt(n int) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = int64(i%997 + 1)
	}
	return p
}

func fixStored(g int, kind string, bs int, h uint64, parent any, toks []int64) Event {
	return Event{Type: "BlockStored", BlockHashes: []any{h}, ParentBlockHash: parent,
		TokenIDs: toks, BlockSize: bs, GroupIdx: &g, Kind: kind}
}

// storeChain stores n full blocks of p in one attention group, hashes from first.
func storeChain(m *Mirror, seq int64, p []int64, bs, n int, first uint64) {
	var evs []Event
	var parent any
	for i := 0; i < n; i++ {
		h := first + uint64(i)
		evs = append(evs, fixStored(0, "full_attention", bs, h, parent, p[i*bs:(i+1)*bs]))
		parent = h
	}
	m.Apply(seq, evs)
}

// Plain vLLM (no --speculative-config) serves every cached block; the drop is opt in
// (review r4-83 #1, r4-87 F2).
func TestFixSpeculativeDropIsOptIn(t *testing.T) {
	p := fixPrompt(4*3200 + 1)
	m := New()
	m.SetGeometry(3200, 0)
	storeChain(m, 0, p, 3200, 4, 1)
	if got := m.CachedTokens(p); got != 12800 {
		t.Fatalf("plain engine: mirror %d, want 12800", got)
	}
	m.SetSpeculative(true)
	if got := m.CachedTokens(p); got != 9600 {
		t.Fatalf("speculative engine: mirror %d, want 9600", got)
	}
}

// The prompt ends table is an LRU, not a FIFO: a conversation whose ends keep being looked
// up (its hold judged, its next turn scored) keeps its unit hit however many other turns
// pass (review r4-86 #5).
func TestFixTailsAreLRU(t *testing.T) {
	m := New()
	m.SetSpeculative(true)
	m.SetGeometry(ubs, unit)
	toks := fixtureTokens(t)[:10260]
	tk := m.NewPromptKeys(toks).Tail()
	m.Apply(0, fullBlocks(toks, 2))
	m.Apply(1, tailEntries(toks, len(toks), 900, 901, uint64(899)))
	if got := m.Cached(tk); got != 10176 {
		t.Fatalf("cached %d, want 10176", got)
	}
	m.mu.Lock()
	for i := uint64(0); i < 2*maxTails; i++ {
		m.rememberLocked(1<<40+i, 0, true, false)
		if i%1000 == 0 {
			m.mu.Unlock()
			m.Cached(tk)
			m.mu.Lock()
		}
	}
	m.mu.Unlock()
	if got := m.Cached(tk); got != 10176 {
		t.Fatalf("after %d other ends: cached %d, want 10176", 2*maxTails, got)
	}
}

// A sequence going back (an engine or publisher restart) ends the subscription so it
// resyncs from a replay, marks the mirror unhealthy until then, and moves the generation
// so admission re-judges its holds (review cv-72 #7).
func TestFixRestartResyncs(t *testing.T) {
	m := New()
	m.SetConnected(true)
	toks := seqPrompt(9)
	m.Apply(5, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	gen := m.Generation()
	if ok := m.Apply(0, nil); ok {
		t.Fatal("a restart did not end the subscription")
	}
	if m.Healthy() || m.Generation() == gen || m.CachedTokens(toks) != 0 {
		t.Fatalf("after a restart: healthy %v, generation %d (was %d), cached %d", m.Healthy(), m.Generation(), gen, m.CachedTokens(toks))
	}
}

// Only full and MLA attention need a contiguous prefix and only mamba keeps checkpoints;
// a sliding-window (or other) group is not modelled, and says so instead of being taken
// for full attention (review r4-83 #7).
func TestFixUnsupportedKindPredictsNothing(t *testing.T) {
	p := fixPrompt(2*16 + 1)
	m := New()
	m.SetGeometry(16, 0)
	storeChain(m, 0, p, 16, 2, 1)
	if got := m.CachedTokens(p); got != 32 {
		t.Fatalf("full attention: cached %d, want 32", got)
	}
	m.Apply(1, []Event{fixStored(1, "sliding_window", 16, 50, nil, p[16:32])})
	if st := m.Status(); st.Unsupported != "sliding_window" || m.CachedTokens(p) != 0 {
		t.Fatalf("sliding window group: status %+v, cached %d", st, m.CachedTokens(p))
	}
}

// The README states the oldest vLLM whose KV events the mirror can read: 0.24.0, where
// events became maps tagged by "type" (review r4-87 F1).
func TestFixREADMEStatesVLLMVersion(t *testing.T) {
	b, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "vLLM engine, version 0.24.0 or later") {
		t.Fatal("README does not state the minimum vLLM version")
	}
}
