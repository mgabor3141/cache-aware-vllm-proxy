package kvmirror

import "testing"

// A prompt whose length is a multiple of the unit: the engine registers its attention
// entry at its end, but bounds its own lookup at num_tokens-1, so only a longer prompt can
// read that entry. A resend of the same prompt reaches the checkpoint a unit lower, where
// no attention entry is held: 0 cached. Block 128, unit 16, speculative decoding.
func TestUnitProbeStopsBeforeTheLastToken(t *testing.T) {
	quietLog(t)
	const bs, u = 128, 16
	all := seqPrompt(321)
	p := all[:320]
	m := New()
	m.SetSpeculative(true)
	m.SetGeometry(bs, u)
	m.NewPromptKeys(p)
	m.Apply(0, []Event{
		stored(5, "mla_attention", bs, 1, nil, p[:bs]),
		stored(5, "mla_attention", bs, 2, uint64(1), p[bs:2*bs]),
	})
	m.Apply(1, []Event{
		stored(0, "mamba", u, 900, uint64(899), p[288:304]),
		stored(5, "mla_attention", u, 901, uint64(900), p[304:320]),
	})
	if got := m.CachedTokens(all); got != 304 {
		t.Fatalf("a longer prompt reads the entry at 320: %d cached, want 304", got)
	}
	if got := m.CachedTokens(p); got != 0 {
		t.Fatalf("a resend of the aligned prompt: %d cached, want 0", got)
	}
	// Admitted keys stand for the next turn, which may read the entry.
	k := m.NewPromptKeys(p).Tail()
	if got := m.Cached(k); got != 304 {
		t.Fatalf("the admitted prompt's tail keys: %d cached, want 304", got)
	}
}
