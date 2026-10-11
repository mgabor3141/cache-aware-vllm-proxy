package kvmirror

import (
	"strings"
	"testing"
)

// Every integer width and bytes msgpack can decode a hash into is read alike; a hash the
// mirror cannot read is counted and logged instead of vanishing.
func TestHashWidthsAndUnreadableHash(t *testing.T) {
	toks := seqPrompt(6)
	for _, h := range []any{uint64(7), int64(7), int(7), uint32(7), int32(7), uint16(7), int16(7), uint8(7), int8(7), []byte{7}} {
		m := New()
		st := attnStore(0, nil, toks[:4])
		st.BlockHashes = []any{h}
		m.Apply(-1, []Event{st})
		if got := m.CachedTokens(toks); got != 4 {
			t.Fatalf("hash %T: %d cached, want 4", h, got)
		}
	}
	logs := quietLog(t)
	m := New()
	st := attnStore(0, nil, toks[:4])
	st.BlockHashes = []any{"0a1b"}
	m.Apply(-1, []Event{st, st})
	if s := m.Status().Stats; s.SkippedHash != 2 || s.Stored != 0 {
		t.Fatalf("stats %+v, want SkippedHash 2 and nothing stored", s)
	}
	if strings.Count(logs.String(), "cannot be read") != 1 {
		t.Fatalf("want one log line, got %q", logs.String())
	}
}
