package kvmirror

import (
	"strings"
	"testing"
)

// Only GPU blocks are mirrored (no medium or "GPU"); offloaded tiers are counted and
// logged, never predicted as cached.
func TestMediumFilter(t *testing.T) {
	toks := seqPrompt(6)
	s := func(v string) *string { return &v }
	for _, c := range []struct {
		medium *string
		want   int
	}{{nil, 4}, {s(""), 4}, {s("GPU"), 4}, {s("CPU"), 0}, {s("DISK"), 0}, {s("cpu"), 0}} {
		logs := quietLog(t)
		m := New()
		st := attnStore(1, nil, toks[:4])
		st.Medium = c.medium
		m.Apply(-1, []Event{st})
		if got := m.CachedTokens(toks); got != c.want {
			t.Fatalf("medium %v: %d cached, want %d", c.medium, got, c.want)
		}
		if c.want == 0 && (m.Status().Stats.SkippedMedium != 1 || !strings.Contains(logs.String(), "offload")) {
			t.Fatalf("medium %q not surfaced: %+v %q", *c.medium, m.Status().Stats, logs.String())
		}
	}
}
