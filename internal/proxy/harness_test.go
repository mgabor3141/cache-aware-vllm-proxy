package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

// logBuf collects the proxy's JSON log lines.
type logBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *logBuf) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *logBuf) events() []map[string]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(l.b.String()), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if json.Unmarshal([]byte(line), &m) == nil {
			out = append(out, m)
		}
	}
	return out
}

// harness runs the proxy as a real HTTP server in front of a stub engine.
type harness struct {
	t    *testing.T
	p    *Proxy
	srv  *httptest.Server
	eng  *httptest.Server
	logs *logBuf
}

const testKey = "test-key"

// engineMux answers /tokenize (three tokens) and /v1/models, and
// hands everything else to inference.
func engineMux(inference http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize":
			io.Copy(io.Discard, r.Body)
			w.Write([]byte(`{"count":3,"tokens":[1,2,3]}`))
		case "/v1/models":
			w.Write([]byte(`{"object":"list","data":[{"id":"served-model","max_model_len":1000}]}`))
		default:
			if inference == nil {
				w.Write([]byte("{}"))
				return
			}
			inference(w, r)
		}
	}
}

func newHarness(t *testing.T, inference http.HandlerFunc, mutate func(*Config)) *harness {
	t.Helper()
	return newHarnessEngine(t, engineMux(inference), mutate)
}

func newTestServer(t *testing.T, h http.Handler) *httptest.Server {
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

// newHarnessEngine is newHarness with a whole engine handler.
func newHarnessEngine(t *testing.T, engine http.Handler, mutate func(*Config)) *harness {
	t.Helper()
	eng := newTestServer(t, engine)
	cfg := &Config{
		Upstream:  eng.URL,
		Auth:      auth.Config{Mode: "static", Keys: []auth.StaticKey{{Key: testKey, User: "alice"}}},
		Admission: AdmissionConfig{MaxActive: 2, Cache: CacheConfig{KVEventsEndpoint: "tcp://127.0.0.1:1"}},
	}
	if mutate != nil {
		mutate(cfg)
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	logs := &logBuf{}
	p, err := NewProxy(cfg, NewEventLogger(logs))
	if err != nil {
		t.Fatal(err)
	}
	p.ready.set(true, "")
	p.admission.SetPool(1000, 16) // as the engine pool loop would; until then nothing is admitted
	srv := httptest.NewServer(p)
	t.Cleanup(srv.Close)
	return &harness{t: t, p: p, srv: srv, eng: eng, logs: logs}
}

func (h *harness) request(ctx context.Context, method, path, body string, hdr map[string]string) (*http.Response, error) {
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, h.srv.URL+path, rd)
	if err != nil {
		h.t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+testKey)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		if v == "" {
			req.Header.Del(k)
		} else {
			req.Header.Set(k, v)
		}
	}
	return http.DefaultClient.Do(req)
}

// do sends a request and returns the status and the whole body.
func (h *harness) do(method, path, body string, hdr map[string]string) (int, string) {
	h.t.Helper()
	resp, err := h.request(context.Background(), method, path, body, hdr)
	if err != nil {
		h.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// lastEvent waits for the n-th log line and returns it.
func (h *harness) event(n int) map[string]any {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if ev := h.logs.events(); len(ev) >= n {
			return ev[n-1]
		}
		time.Sleep(5 * time.Millisecond)
	}
	h.t.Fatalf("no log line %d; have %v", n, h.logs.events())
	return nil
}

// waitFor polls cond for up to 5 s.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func num(v any) int {
	f, _ := v.(float64)
	return int(f)
}

const chatBody = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`

// realVLLMChatResponse is a vLLM ChatCompletionResponse: fields follow usage.
const realVLLMChatResponse = `{"id":"chatcmpl-9","object":"chat.completion","created":1728540000,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"391"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":78,"completion_tokens":5,"total_tokens":83,"prompt_tokens_details":{"cached_tokens":64}},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}`

func intPtr(i int) *int { return &i }
