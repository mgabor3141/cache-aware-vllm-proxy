package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseResponseUsage(t *testing.T) {
	u, ok := parseResponseUsage([]byte(realVLLMChatResponse))
	if !ok || u.Prompt != 78 || u.Completion != 5 || u.Total != 83 || u.Cached != 64 || !u.Details {
		t.Fatalf("%+v %v", u, ok)
	}
	// /v1/completions puts kv_transfer_params after usage too.
	u, ok = parseResponseUsage([]byte(`{"choices":[{"text":"391"}],"usage":{"prompt_tokens":78,"total_tokens":83,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":0}},"kv_transfer_params":null}`))
	if !ok || u.Prompt != 78 || u.Cached != 0 || !u.Details {
		t.Fatalf("completions: %+v %v", u, ok)
	}
	for _, body := range []string{
		`{"choices":[]}`, `{"usage":null}`, `{"usage":{"total_tokens":0}}`, `not json`,
		// A model writing a usage object as text is not usage.
		`{"choices":[{"message":{"content":"\"usage\":{\"prompt_tokens\":5,\"total_tokens\":9}"}}]}`,
	} {
		if u, ok := parseResponseUsage([]byte(body)); ok {
			t.Errorf("%s: %+v", body, u)
		}
	}
}

// A real miss (details, 0 cached) and a blind spot (no details) log differently.
func TestUsageDetailsSeparateMissFromBlindSpot(t *testing.T) {
	miss, _ := parseResponseUsage([]byte(`{"usage":{"prompt_tokens":9,"total_tokens":10,"prompt_tokens_details":{"cached_tokens":0}}}`))
	blind, _ := parseResponseUsage([]byte(`{"usage":{"prompt_tokens":9,"total_tokens":10}}`))
	line := func(u tokenUsage) string {
		ev := &Event{}
		ev.setUsage(u)
		b, _ := json.Marshal(ev)
		return string(b)
	}
	if l := line(miss); !strings.Contains(l, `"usage_details":true`) || strings.Contains(l, "cached_prompt_tokens") {
		t.Errorf("miss: %s", l)
	}
	if l := line(blind); strings.Contains(l, "usage_details") {
		t.Errorf("blind spot: %s", l)
	}
}

func TestUsageScanner(t *testing.T) {
	chunks := (&AdmissionQueue{cfg: AdmissionConfig{Cache: CacheConfig{TokensPerChunk: 4}}}).NewOutputCounter()
	s := &usageScanner{chunks: chunks}
	stream := sseChunk + ": keepalive\n\n" +
		// Spacing must not matter for chunk counting (cv-59 #13).
		`data: { "choices" : [ {"index":0,"delta":{"content":"y"}} ], "usage": null }` + "\n\n"
	// Byte by byte: lines split across writes.
	for i := 0; i < len(stream); i++ {
		s.Write([]byte{stream[i]})
	}
	if chunks.Load() != 8 {
		t.Fatalf("two choice chunks counted %d, want 2x4", chunks.Load())
	}
	s.Write([]byte(sseUsage + sseDone))
	if !s.found || s.usage.Prompt != 40 || s.usage.Cached != 32 || !s.done {
		t.Fatalf("%+v", s)
	}
	// A usage chunk replaces the estimate with the engine's count.
	if chunks.Load() != 2 {
		t.Fatalf("counted %d", chunks.Load())
	}
}

func TestUsageScannerNoUsageNoDone(t *testing.T) {
	s := &usageScanner{}
	s.Write([]byte(sseChunk + sseChunk))
	if s.found || s.done {
		t.Fatalf("%+v", s)
	}
}

// An over-long line is skipped, not buffered without bound, and scanning resumes after it.
func TestUsageScannerSkipsOverlongLine(t *testing.T) {
	s := &usageScanner{}
	long := "data: " + strings.Repeat("x", maxSSELine+10)
	for i := 0; i < len(long); i += 4096 {
		s.Write([]byte(long[i:min(i+4096, len(long))]))
	}
	if len(s.pending) > maxSSELine {
		t.Fatalf("pending grew to %d", len(s.pending))
	}
	s.Write([]byte("\n\n" + sseUsage + sseDone))
	if !s.found || !s.done {
		t.Fatalf("%+v", s.usage)
	}
}

func TestEnsureStreamUsage(t *testing.T) {
	for _, c := range []struct {
		in   string
		want string
	}{
		{`{}`, `{"stream_options":{"include_usage":true}}`},
		{`{"stream_options":{"include_usage":false}}`, `{"stream_options":{"include_usage":false}}`},
		{`{"stream_options":{"continuous_usage_stats":true}}`, `{"stream_options":{"continuous_usage_stats":true,"include_usage":true}}`},
	} {
		var b map[string]any
		json.Unmarshal([]byte(c.in), &b)
		ensureStreamUsage(b)
		if out, _ := json.Marshal(b); string(out) != c.want {
			t.Errorf("%s: %s, want %s", c.in, out, c.want)
		}
	}
}
