package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
)

// tokenizeProxy is a Proxy whose engine answers /tokenize with tokenize, recording
// the request.
func tokenizeProxy(t *testing.T, tokenize http.HandlerFunc) (*Proxy, *atomic.Value) {
	t.Helper()
	seen := &atomic.Value{}
	up := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		seen.Store(body)
		tokenize(w, r)
	}))
	u, _ := url.Parse(up.URL)
	q := NewAdmissionQueue(AdmissionConfig{Cache: CacheConfig{KVEventsEndpoint: "tcp://127.0.0.1:1"}})
	q.SetPool(1000, 4, 0)
	return &Proxy{upstream: u, admission: q, tokenizeClient: &http.Client{}, tokenizeSem: make(chan struct{}, 4)}, seen
}

// parsedEvent runs a body through the same parsing serveInference does.
func parsedEvent(t *testing.T, body string) *Event {
	t.Helper()
	var m map[string]any
	dec := json.NewDecoder(strings.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&m); err != nil {
		t.Fatal(err)
	}
	return &Event{body: m, maxTokens: maxTokensOf(m), textBytes: textBytesOf(m)}
}

func chatRequest() *http.Request {
	return httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
}

func TestAdmissionRequestTokenizesWithTemplateKwargs(t *testing.T) {
	p, seen := tokenizeProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"count":10,"tokens":[1,2,3,4,5,6,7,8,9,10]}`))
	})
	ev := parsedEvent(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}],"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"none","max_tokens":50}`)
	ar := p.admissionRequest(context.Background(), chatRequest(), ev)
	if ar.PromptTokens != 10 || !ev.Tokenized || ar.MaxTokens != 50 {
		t.Fatalf("ar=%+v ev.Tokenized=%v", ar, ev.Tokenized)
	}
	sent := seen.Load().(map[string]any)
	for _, k := range []string{"messages", "tools", "chat_template_kwargs", "add_generation_prompt"} {
		if _, ok := sent[k]; !ok {
			t.Errorf("tokenize request lacks %s: %v", k, sent)
		}
	}
	if kw := sent["chat_template_kwargs"].(map[string]any); kw["reasoning_effort"] != "none" || kw["enable_thinking"] != false {
		t.Errorf("template kwargs %v", kw)
	}
	if ev.body != nil {
		t.Error("body kept after tokenizing")
	}
	if len(ar.Tokens) != 10 {
		t.Errorf("tokens %v, want the 10 /tokenize returned", ar.Tokens)
	}
}

// A 4xx from /tokenize never charges less than the estimate.
func TestAdmissionRequestTokenizeRejected(t *testing.T) {
	p, _ := tokenizeProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"At most 2 image(s)"}`, http.StatusBadRequest)
	})
	ev := parsedEvent(t, `{"model":"m","messages":[{"role":"user","content":"`+strings.Repeat("x", 400000)+`"}]}`)
	ar := p.admissionRequest(context.Background(), chatRequest(), ev)
	if ar.PromptTokens < 90000 || ev.TokenizeError == "" {
		t.Fatalf("prompt %d err %q, want the ~100K byte estimate and an error", ar.PromptTokens, ev.TokenizeError)
	}
}

// Engine failing: fall back to the body's bytes; an inline image counts as
// fallbackMediaTokens, not its base64 size and not nothing (chaos #5).
func TestAdmissionRequestTokenizeFailsFallsBackToTextBytes(t *testing.T) {
	p, _ := tokenizeProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	img := "data:image/png;base64," + strings.Repeat("A", 1000000)
	ev := parsedEvent(t, `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"`+strings.Repeat("y", 40000)+`"},{"type":"image_url","image_url":{"url":"`+img+`"}}]}]}`)
	ar := p.admissionRequest(context.Background(), chatRequest(), ev)
	if want := 10000 + fallbackMediaTokens; ar.PromptTokens < want-500 || ar.PromptTokens > want+2000 {
		t.Fatalf("fallback prompt %d, want about %d (40 KB of text and one image)", ar.PromptTokens, want)
	}
}

// A string that merely starts with "data:" is text: it prices at its bytes, so a caller
// cannot make a large prompt free by prefixing it (attacker #3).
func TestFallbackPricesDataPrefixedTextAsText(t *testing.T) {
	p, _ := tokenizeProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	ev := parsedEvent(t, `{"model":"m","messages":[{"role":"user","content":"data:`+strings.Repeat("z", 400000)+`"}]}`)
	ar := p.admissionRequest(context.Background(), chatRequest(), ev)
	if ar.PromptTokens < 100000 {
		t.Fatalf("a 400 KB prompt starting with data: priced at %d tokens", ar.PromptTokens)
	}
}

// /v1/completions tokenizes its prompt.
func TestAdmissionRequestCompletionsPrompt(t *testing.T) {
	p, seen := tokenizeProxy(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"tokens":[1,2]}`)
	})
	ev := parsedEvent(t, `{"model":"m","prompt":"hello"}`)
	ar := p.admissionRequest(context.Background(), httptest.NewRequest("POST", "/v1/completions", nil), ev)
	if ar.PromptTokens != 2 || seen.Load().(map[string]any)["prompt"] != "hello" {
		t.Fatalf("%+v %v", ar, seen.Load())
	}
}

func TestSessionKeyIncludesCallerAndModel(t *testing.T) {
	body := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	if sessionKey("m", "static/alice", body) == sessionKey("m", "static/bob", body) {
		t.Fatal("two callers share a session")
	}
	if sessionKey("m", "static/alice", body) == sessionKey("n", "static/alice", body) {
		t.Fatal("two models share a session")
	}
	if sessionKey("m", "a", map[string]any{}) != "" {
		t.Fatal("a session without messages")
	}
}

func TestMaxTokensOf(t *testing.T) {
	ev := parsedEvent(t, `{"max_tokens":900}`)
	if ev.maxTokens != 900 {
		t.Fatal(ev.maxTokens)
	}
	ev = parsedEvent(t, `{"max_completion_tokens":50,"max_tokens":900}`)
	if ev.maxTokens != 50 {
		t.Fatal(ev.maxTokens)
	}
}
