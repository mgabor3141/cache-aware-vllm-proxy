package kvmirror

import (
	"testing"
)

// P1: hybrid model, full-mode report for the mamba group lists every block 0..k.
func TestFixRevMambaReportFalseHit(t *testing.T) {
	bs := 4
	p := seqPrompt(4*bs + 1)
	m := New()
	var evs []Event
	var parent any
	for i := 0; i < 4; i++ {
		evs = append(evs, stored(0, "full_attention", bs, uint64(i+1), parent, p[i*bs:(i+1)*bs]))
		parent = uint64(i + 1)
	}
	// mamba state only at the end of block 4 (hash 4)
	evs = append(evs, stored(1, "mamba", bs, 4, uint64(3), p[3*bs:4*bs]))
	m.Apply(0, evs)
	short := p[:2*bs+1]
	before := m.CachedTokens(short)
	g := 1
	m.Apply(1, []Event{{Type: "BlockStored", BlockHashes: []any{uint64(1), uint64(2), uint64(3), uint64(4)},
		TokenIDs: p[:4*bs], BlockSize: bs, GroupIdx: &g, Kind: "mamba"}})
	after := m.CachedTokens(short)
	t.Logf("short prompt cached before report=%d after=%d stats=%+v", before, after, m.Status().Stats)
	if after != before {
		t.Fatalf("report changed prediction %d -> %d (false hit)", before, after)
	}
}

// P3: report covering a block the mirror missed re-adds copies of held blocks.
func TestFixRevPartialReportDoubleCount(t *testing.T) {
	bs := 4
	p := seqPrompt(3*bs + 1)
	m := New()
	m.Apply(0, []Event{attnStore(1, nil, p[:bs]), attnStore(2, uint64(1), p[bs:2*bs])})
	g := 5
	// engine holds 1,2,3 (3's store missed); report all three
	m.Apply(1, []Event{{Type: "BlockStored", BlockHashes: []any{uint64(1), uint64(2), uint64(3)},
		TokenIDs: p[:3*bs], BlockSize: bs, GroupIdx: &g, Kind: "full_attention"}})
	m.Apply(2, []Event{removed(5, 1), removed(5, 2), removed(5, 3)})
	if got := m.CachedTokens(p); got != 0 {
		t.Fatalf("all evicted, mirror still says %d cached", got)
	}
}
