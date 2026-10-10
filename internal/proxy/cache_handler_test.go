package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

func cacheTestProxy(t *testing.T, tokenize http.HandlerFunc) (*Proxy, *atomic.Value) {
	t.Helper()
	seen := &atomic.Value{}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/tokenize" {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			seen.Store(body)
			tokenize(w, r)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(up.Close)
	u, _ := url.Parse(up.URL)
	cfg := &Config{
		Aliases: map[string]AliasEntry{"default": {Target: "m"}},
		Admission: AdmissionConfig{MaxActive: 8,
			Cache: CacheConfig{KVEventsEndpoint: "tcp://vllm:5557"}},
	}
	cfg.Admission.applyDefaults()
	q := NewAdmissionQueue(cfg.Admission)
	q.SetPool(1000, 4)
	return &Proxy{cfg: cfg, upstream: u, tokenizeClient: &http.Client{}, tokenizeSem: make(chan struct{}, 4), admission: q}, seen
}

func rewriteForAdmission(t *testing.T, p *Proxy, body string, hdr map[string]string) (*http.Request, *Event) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	ev := &Event{}
	if ok, _ := p.rewriteBody(httptest.NewRecorder(), req, &auth.Result{User: "u", Source: "static", Priority: 0}, ev); !ok {
		t.Fatal("rewrite failed")
	}
	return req, ev
}

func TestAdmissionRequestTokenizesWithTemplateKwargs(t *testing.T) {
	p, seen := cacheTestProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"count":10,"tokens":[1,2,3,4,5,6,7,8,9,10]}`))
	})
	req, ev := rewriteForAdmission(t, p, `{"model":"default","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}],"chat_template_kwargs":{"enable_thinking":false},"max_tokens":50}`,
		map[string]string{"X-Priority": "-50"})
	ar := p.admissionRequest(req, ev)
	if ar.PromptTokens != 10 || len(ar.Blocks.Blocks) != 2 || !ev.Tokenized || ar.MaxTokens != 50 {
		t.Fatalf("ar=%+v ev.Tokenized=%v", ar, ev.Tokenized)
	}
	sent := seen.Load().(map[string]any)
	for _, k := range []string{"messages", "tools", "chat_template_kwargs", "add_generation_prompt"} {
		if _, ok := sent[k]; !ok {
			t.Errorf("tokenize request lacks %s: %v", k, sent)
		}
	}
	if ev.Priority != 0 {
		t.Errorf("priority %d: a caller ran sooner than its identity allows (0)", ev.Priority)
	}
	if req.Header.Get("X-Priority") != "" {
		t.Error("X-Priority would be forwarded upstream")
	}
	if ev.admitBody != nil {
		t.Error("body kept after tokenizing")
	}
}

// A 4xx from /tokenize never charges less than the estimate.
func TestAdmissionRequestTokenizeRejected(t *testing.T) {
	p, _ := cacheTestProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"At most 2 image(s)"}`, http.StatusBadRequest)
	})
	req, ev := rewriteForAdmission(t, p, `{"model":"default","messages":[{"role":"user","content":"`+strings.Repeat("x", 400000)+`"}]}`, nil)
	ar := p.admissionRequest(req, ev)
	if ar.PromptTokens < 90000 || ev.TokenizeError == "" {
		t.Fatalf("prompt %d err %q, want the ~100K byte estimate and an error", ar.PromptTokens, ev.TokenizeError)
	}
}

// Engine unreachable or failing: fall back to text bytes, ignoring inline media.
func TestAdmissionRequestTokenizeFailsFallsBackToTextBytes(t *testing.T) {
	p, _ := cacheTestProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	img := "data:image/png;base64," + strings.Repeat("A", 1000000)
	req, ev := rewriteForAdmission(t, p, `{"model":"default","messages":[{"role":"user","content":[{"type":"text","text":"`+strings.Repeat("y", 40000)+`"},{"type":"image_url","image_url":{"url":"`+img+`"}}]}]}`, nil)
	ar := p.admissionRequest(req, ev)
	if ar.PromptTokens < 9000 || ar.PromptTokens > 12000 {
		t.Fatalf("fallback prompt %d, want about 10K (40 KB of text, image ignored)", ar.PromptTokens)
	}
}

// Fingerprint sessions include the caller, so two users with the same opening differ.
func TestSessionKeyIncludesCaller(t *testing.T) {
	body := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	if sessionKey("m", "static/alice/", body) == sessionKey("m", "static/bob/", body) {
		t.Fatal("two callers share a fingerprint session")
	}
}

// A caller may ask to run later than its identity's priority, never sooner.
func TestCallerPriorityOnlyLater(t *testing.T) {
	p, _ := cacheTestProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"count":1,"tokens":[1]}`))
	})
	for hdr, want := range map[string]int{"7": 7, "-3": 0, "500": 100} {
		_, ev := rewriteForAdmission(t, p, `{"model":"default","messages":[{"role":"user","content":"hi"}]}`,
			map[string]string{"X-Priority": hdr})
		if ev.Priority != want {
			t.Errorf("X-Priority %s: priority %d, want %d", hdr, ev.Priority, want)
		}
	}
}

// An alias override cannot make a caller more urgent than its identity either.
func TestAliasOverrideOnlyLater(t *testing.T) {
	p, _ := cacheTestProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"count":1,"tokens":[1]}`))
	})
	ten := 10
	p.cfg.Aliases = map[string]AliasEntry{"batch": {Target: "m", PriorityOverride: &ten}}
	for _, c := range []struct{ base, want int }{{0, 10}, {20, 20}} {
		body := `{"model":"batch","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		ev := &Event{}
		if ok, _ := p.rewriteBody(httptest.NewRecorder(), req, &auth.Result{User: "u", Source: "static", Priority: c.base}, ev); !ok {
			t.Fatal("rewrite failed")
		}
		if ev.Priority != c.want {
			t.Errorf("identity %d via batch: priority %d, want %d", c.base, ev.Priority, c.want)
		}
	}
}
