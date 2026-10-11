package kvmirror

import (
	"context"
	"testing"
	"time"
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

// P2: a replay hole that held a BlockRemoved leaves a block the engine dropped.
func TestFixRevReplayHoleLosesRemoval(t *testing.T) {
	old := replayFirstWait
	replayFirstWait = 300 * time.Millisecond
	defer func() { replayFirstWait = old }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	toks := seqPrompt(9)
	other := seqPrompt(40)[20:24]
	buf := []rawBatch{
		{1, batchPayload(attnStore(1, nil, toks[:4]))},
		{2, batchPayload(attnStore(2, uint64(1), toks[4:8]))},
		// 3: BlockRemoved 1,2: lost from the buffer
		{4, batchPayload(attnStore(9, nil, other))},
	}
	m := New()
	if _, err := m.replay(ctx, serveReplay(ctx, t, buf)); err != nil {
		t.Fatal(err)
	}
	if got := m.CachedTokens(toks); got != 0 {
		t.Fatalf("engine removed blocks 1,2 in the hole; mirror says %d cached", got)
	}
}
