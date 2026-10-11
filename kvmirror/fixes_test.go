package kvmirror

import "testing"

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

// With kv_cache_report_mode "full" (settable per request through vllm_xargs) vLLM
// publishes a BlockStored for every block a request reused, parent nil, from block 0,
// without a matching BlockRemoved. Those must not count as copies (review r4-83 #4).
func TestFixFullReportIsNotACopy(t *testing.T) {
	p := fixPrompt(3*3200 + 1)
	m := New()
	m.SetGeometry(3200, 0)
	storeChain(m, 0, p, 3200, 3, 1)
	g := 0
	m.Apply(1, []Event{{Type: "BlockStored", BlockHashes: []any{uint64(1), uint64(2), uint64(3)},
		TokenIDs: p[:3*3200], BlockSize: 3200, GroupIdx: &g, Kind: "full_attention"}})
	m.Apply(2, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(1), uint64(2), uint64(3)}, GroupIdx: &g}})
	if st := m.Status(); st.Blocks != 0 || m.CachedTokens(p) != 0 {
		t.Fatalf("%d blocks still cached after their only eviction", st.Blocks)
	}
}
