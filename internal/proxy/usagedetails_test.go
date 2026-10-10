package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// The three states that the access log used to collapse into one. A turn whose engine reported
// prompt_tokens_details with cached_tokens 0 is a real full miss; a turn whose engine reported no
// details at all is a blind spot. CachedPromptTokens is serialised with omitempty, so both produce
// a JSON line with no cached_prompt_tokens key, and on 2026-10-02 a reuse rate of 97.6% was reported
// from data whose honest range was 86.4% to 58.5% depending on which of those two was assumed.
//
// This test pins the parser and the serialisation together, because the bug lived in the seam
// between them: the parser could already tell (PromptDetails is a pointer) and the record could not.
func TestUsageDetails_SeparatesRealMissFromBlindSpot(t *testing.T) {
	miss := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"391"}}],
		"usage":{"prompt_tokens":114275,"total_tokens":114291,"completion_tokens":16,
		"prompt_tokens_details":{"cached_tokens":0}}}`)
	blind := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"391"}}],
		"usage":{"prompt_tokens":114275,"total_tokens":114291,"completion_tokens":16}}`)

	uMiss, ok := parseResponseUsage(miss)
	if !ok {
		t.Fatal("expected the miss body to yield usage")
	}
	if !uMiss.Details {
		t.Error("Details = false for a response that carried prompt_tokens_details; the whole point of the field")
	}
	if uMiss.Cached != 0 {
		t.Errorf("Cached = %d, want 0 for a full miss", uMiss.Cached)
	}

	uBlind, ok := parseResponseUsage(blind)
	if !ok {
		t.Fatal("expected the blind-spot body to yield usage")
	}
	if uBlind.Details {
		t.Error("Details = true for a response with no prompt_tokens_details; that would record a blind spot as a measured miss")
	}
	if uBlind.Cached != 0 {
		t.Errorf("Cached = %d, want 0 when nothing was reported", uBlind.Cached)
	}

	// The seam. Both cases write CachedPromptTokens 0, so the record alone must carry the
	// difference or the log line is identical and the ambiguity is back.
	missLine := marshalRecord(t, Event{PromptTokens: 114275, CachedPromptTokens: uMiss.Cached, UsageDetails: uMiss.Details})
	blindLine := marshalRecord(t, Event{PromptTokens: 114275, CachedPromptTokens: uBlind.Cached, UsageDetails: uBlind.Details})

	if !strings.Contains(missLine, `"usage_details":true`) {
		t.Errorf("a real miss must be marked in the log, got %s", missLine)
	}
	if strings.Contains(missLine, "cached_prompt_tokens") {
		t.Errorf("omitempty should still drop the zero value, got %s", missLine)
	}
	if strings.Contains(blindLine, "usage_details") {
		t.Errorf("a blind spot must not claim usage_details, got %s", blindLine)
	}
	if missLine == blindLine {
		t.Fatal("the two records serialise identically; the ambiguity this field exists to remove is still present")
	}
}

// And the state that must stay absent rather than become a zero: a request with no usage at all,
// which is every /v1/models poll and every request without usage. Emitting cached_prompt_tokens:0 for those
// would invent misses that never happened, which is why the number keeps omitempty.
func TestUsageDetails_NoUsageStaysUnmarked(t *testing.T) {
	u, ok := parseResponseUsage([]byte(`{"object":"list","data":[{"id":"qwen3.8-flash-next-w4"}]}`))
	if ok || u.Details {
		t.Fatalf("a response without usage must not report details, ok=%v details=%v", ok, u.Details)
	}
	line := marshalRecord(t, Event{Path: "/v1/models", PromptTokens: u.Prompt,
		CachedPromptTokens: u.Cached, UsageDetails: u.Details})
	for _, bad := range []string{"cached_prompt_tokens", "usage_details", "prompt_tokens"} {
		if strings.Contains(line, bad) {
			t.Errorf("a usage-free record must not carry %s, got %s", bad, line)
		}
	}
}

func marshalRecord(t *testing.T, ev Event) string {
	t.Helper()
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}
