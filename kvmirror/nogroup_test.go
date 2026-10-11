package kvmirror

import (
	"strings"
	"testing"
)

// An event without a usable group_idx is counted and logged, never silent; a removal
// without one forgets the hash in every group (under-reporting), and a group of 64 or
// more, which the masks cannot hold, is treated the same.
func TestEventWithoutGroupIsSurfaced(t *testing.T) {
	logs := quietLog(t)
	toks := seqPrompt(9)
	m := New()
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	if got := m.CachedTokens(toks); got != 8 {
		t.Fatalf("setup: %d cached, want 8", got)
	}
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(2)}}})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("a removal without group_idx: %d cached, want 4", got)
	}
	big := 70
	st := attnStore(3, uint64(1), toks[4:8])
	st.GroupIdx, st.BlockHashes = &big, []any{uint64(3)}
	noGroup := attnStore(4, nil, toks[:4])
	noGroup.GroupIdx = nil
	m.Apply(-1, []Event{st, noGroup})
	if s := m.Status().Stats; s.SkippedNoGroup != 2 {
		t.Fatalf("SkippedNoGroup %d, want 2", s.SkippedNoGroup)
	}
	if !strings.Contains(logs.String(), "group_idx") {
		t.Fatalf("not logged: %q", logs.String())
	}
}
