package proxy

import "testing"

func TestParseResponseUsage_NonStreaming(t *testing.T) {
	body := []byte(`{"id":"x","choices":[{"message":{"content":"hi"}}],
		"usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}`)
	u, ok := parseResponseUsage(body)
	if !ok {
		t.Fatal("expected usage found")
	}
	if u.Prompt != 12 || u.Completion != 34 || u.Total != 46 {
		t.Errorf("got %+v, want {12 34 46}", u)
	}
}

func TestParseResponseUsage_Missing(t *testing.T) {
	for _, body := range [][]byte{
		[]byte(`{"id":"x","choices":[]}`),      // no usage
		[]byte(`{"usage":null}`),               // null usage
		[]byte(`{"usage":{"total_tokens":0}}`), // zero total -> not usable
		[]byte(`not json at all`),              // garbage
		[]byte(``),                             // empty
	} {
		if _, ok := parseResponseUsage(body); ok {
			t.Errorf("expected no usage for %q", body)
		}
	}
}

func TestUsageScanner_FindsFinalChunk(t *testing.T) {
	s := &usageScanner{}
	// Typical vLLM include_usage stream: content chunks with usage:null,
	// then a final empty-choices chunk carrying usage, then [DONE].
	s.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}],\"usage\":null}\n\n"))
	s.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}],\"usage\":null}\n\n"))
	s.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":7,\"total_tokens\":107}}\n\n"))
	s.Write([]byte("data: [DONE]\n\n"))
	u, ok := s.Usage()
	if !ok {
		t.Fatal("expected usage found")
	}
	if u.Prompt != 100 || u.Completion != 7 || u.Total != 107 {
		t.Errorf("got %+v, want {100 7 107}", u)
	}
}

func TestUsageScanner_SplitAcrossWrites(t *testing.T) {
	s := &usageScanner{}
	full := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n"
	// Feed one byte at a time to prove line reassembly across Write calls.
	for i := 0; i < len(full); i++ {
		s.Write([]byte{full[i]})
	}
	u, ok := s.Usage()
	if !ok || u.Total != 8 {
		t.Fatalf("got %+v ok=%v, want total 8", u, ok)
	}
}

func TestUsageScanner_NoUsageInStream(t *testing.T) {
	s := &usageScanner{}
	s.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
	s.Write([]byte("data: [DONE]\n\n"))
	if _, ok := s.Usage(); ok {
		t.Error("expected no usage when stream omits it")
	}
}

func TestEnsureStreamUsage_InjectsWhenMissing(t *testing.T) {
	body := map[string]any{"model": "x", "stream": true}
	ensureStreamUsage(body)
	so, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatal("stream_options not created")
	}
	if so["include_usage"] != true {
		t.Errorf("include_usage=%v, want true", so["include_usage"])
	}
}

func TestEnsureStreamUsage_RespectsCallerValue(t *testing.T) {
	// Caller explicitly disabled it: do not override.
	body := map[string]any{
		"stream_options": map[string]any{"include_usage": false},
	}
	ensureStreamUsage(body)
	so := body["stream_options"].(map[string]any)
	if so["include_usage"] != false {
		t.Errorf("include_usage=%v, want false (caller wins)", so["include_usage"])
	}
}

func TestEnsureStreamUsage_PreservesOtherStreamOptions(t *testing.T) {
	body := map[string]any{
		"stream_options": map[string]any{"foo": "bar"},
	}
	ensureStreamUsage(body)
	so := body["stream_options"].(map[string]any)
	if so["foo"] != "bar" {
		t.Error("existing stream_options dropped")
	}
	if so["include_usage"] != true {
		t.Error("include_usage not added alongside existing options")
	}
}

// The prefix-cache hit is the number the whole KV-thrash question turns on, so
// its decoding is worth pinning to the shapes the engine actually emits. These
// payloads are copied from the live vLLM chat completion response, not invented:
// details present with a real hit, and details absent on paths that do not
// report them. "Not reported" must not be turned into a confident zero-by-magic;
// it arrives as 0 with ok=true and is distinguishable from a hit only because the
// wire field is a pointer, so the distinction lives in usage.go where it belongs.
func TestParseResponseUsage_ReadsCachedPromptTokens(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"qwen3.8-flash-next-w4",
		"choices":[{"index":0,"message":{"role":"assistant","content":"391"}}],"usage":{"prompt_tokens":48211,
		"total_tokens":48266,"completion_tokens":55,"prompt_tokens_details":{"cached_tokens":45056}}}`)

	u, ok := parseResponseUsage(body)
	if !ok {
		t.Fatalf("expected usage to parse from a chat completion carrying usage")
	}
	if got, want := u.Prompt, 48211; got != want {
		t.Errorf("Prompt = %d, want %d", got, want)
	}
	if got, want := u.Cached, 45056; got != want {
		t.Errorf("Cached = %d, want %d", got, want)
	}
	// The recompute bill for the turn. This arithmetic is the measurement, so it
	// is asserted rather than left to the dashboard query.
	if got, want := u.Prompt-u.Cached, 3155; got != want {
		t.Errorf("recomputed prompt tokens = %d, want %d", got, want)
	}
}

// Renamed on 2026-10-02. It used to be TestParseResponseUsage_NoDetailsMeansZeroCached, which
// asserted the same false equivalence that produced a wrong conclusion: Cached stays 0 when the
// engine reports no details because there is nothing to fill it in with, but 0 is also what a real
// full miss looks like, so "no details" never means "measured zero". The parser behaviour is
// unchanged and correct; the name was the bug, and Details is what keeps it from being read that way
// again.
func TestParseResponseUsage_NoDetailsIsABlindSpotNotAMeasuredZero(t *testing.T) {
	body := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}],
		"usage":{"prompt_tokens":100,"total_tokens":120,"completion_tokens":20}}`)

	u, ok := parseResponseUsage(body)
	if !ok {
		t.Fatalf("expected usage to parse when prompt_tokens_details is absent")
	}
	if u.Cached != 0 {
		t.Errorf("Cached = %d, want 0 when the engine reported no details", u.Cached)
	}
	if u.Details {
		t.Error("Details = true with no prompt_tokens_details; a blind spot must not look measured")
	}
	if u.Prompt != 100 {
		t.Errorf("Prompt = %d, want 100", u.Prompt)
	}
}

func TestParseUsageFromTail_JSONCompletion(t *testing.T) {
	// vLLM puts usage after choices, so it is exactly what survives at the end.
	tail := []byte(`{"id":"chatcmpl-9","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"391"},"finish_reason":"stop"}],"usage":{"prompt_tokens":48211,"total_tokens":48266,"completion_tokens":55,"prompt_tokens_details":{"cached_tokens":45056}}}`)
	u, ok := parseUsageFromTail(tail)
	if !ok {
		t.Fatalf("no usage parsed from a completion whose tail carries it")
	}
	if u.Prompt != 48211 || u.Completion != 55 || u.Cached != 45056 {
		t.Errorf("parsed %+v, want prompt 48211 completion 55 cached 45056", u)
	}
}

func TestParseUsageFromTail_StreamFinalChunk(t *testing.T) {
	tail := []byte("data: {\"id\":\"chatcmpl-2\",\"choices\":[],\"usage\":{\"prompt_tokens\":912,\"total_tokens\":940,\"completion_tokens\":28,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n")
	u, ok := parseUsageFromTail(tail)
	if !ok {
		t.Fatalf("no usage parsed from a final SSE chunk")
	}
	if u.Prompt != 912 || u.Cached != 0 {
		t.Errorf("parsed %+v, want prompt 912 cached 0", u)
	}
}

func TestParseUsageFromTail_IgnoresUsageWrittenByTheModel(t *testing.T) {
	// The reason the guards exist: a model explaining token accounting in prose can
	// emit the literal text. It appears inside a string and is followed by more
	// text, so neither the position nor the value is credible.
	tail := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"the field looks like \"usage\":{\"prompt_tokens\":999999,\"total_tokens\":1000000,\"completion_tokens\":1} and means"}}]}`)
	if u, ok := parseUsageFromTail(tail); ok {
		t.Errorf("parsed %+v from prose that only mentions usage; want no match", u)
	}
}

func TestParseUsageFromTail_NoUsageNoMatch(t *testing.T) {
	if _, ok := parseUsageFromTail([]byte(`{"data":[{"id":"tok-1","tokens":[1,2,3]}]}`)); ok {
		t.Error("parsed usage from a /tokenize response that has none")
	}
}
