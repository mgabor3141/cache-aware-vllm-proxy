package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

// Only the allowlisted routes reach the engine; everything else is 404 without a byte
// sent upstream, including engine control routes and paths that only look like an
// allowed one (cv-56 #1, #8).
func TestUnknownRoutesAre404(t *testing.T) {
	var hits atomic.Int32
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }, nil)
	for _, c := range []struct{ method, path string }{
		{"POST", "/v1/embeddings"}, {"POST", "/v1/responses"}, {"GET", "/health"},
		{"POST", "/sleep"}, {"POST", "/collective_rpc"}, {"GET", "/v1/chat/completions"},
		{"DELETE", "/v1/models/x"}, {"POST", "/v1/load_lora_adapter"}, {"GET", "/debug/residency"},
		{"POST", "/%2e/v1/chat/completions"}, {"OPTIONS", "/v1/chat/completions"},
	} {
		if code, _ := h.do(c.method, c.path, chatBody, nil); code != http.StatusNotFound {
			t.Errorf("%s %s: %d, want 404", c.method, c.path, code)
		}
	}
	if hits.Load() != 0 {
		t.Fatalf("the engine saw %d requests", hits.Load())
	}
	if ev := h.event(1); num(ev["status"]) != 404 {
		t.Fatalf("404 not logged: %v", ev)
	}
}

func TestAPIRoutesNeedCredentials(t *testing.T) {
	var hits atomic.Int32
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }, nil)
	for _, c := range []struct{ method, path string }{
		{"GET", "/v1/models"}, {"POST", "/v1/chat/completions"}, {"POST", "/v1/completions"},
		{"POST", "/v1/tokenize"}, {"POST", "/tokenize"},
	} {
		for _, bearer := range []string{"", "Bearer wrong"} {
			if code, _ := h.do(c.method, c.path, chatBody, map[string]string{"Authorization": bearer}); code != http.StatusUnauthorized {
				t.Errorf("%s %s with %q: %d, want 401", c.method, c.path, bearer, code)
			}
		}
	}
	if hits.Load() != 0 {
		t.Fatalf("the engine saw %d requests", hits.Load())
	}
}

// /healthz, /readyz and /metrics need no key and name nothing about callers.
func TestKeylessEndpoints(t *testing.T) {
	h := newHarness(t, nil, nil)
	none := map[string]string{"Authorization": ""}
	if code, _ := h.do("GET", "/healthz", "", none); code != 200 {
		t.Errorf("/healthz %d", code)
	}
	if code, _ := h.do("GET", "/readyz", "", none); code != 200 {
		t.Errorf("/readyz %d while ready", code)
	}
	h.p.ready.set(false, "test")
	if code, _ := h.do("GET", "/readyz", "", none); code != 503 {
		t.Errorf("/readyz %d while not ready", code)
	}
	if code, body := h.do("GET", "/metrics", "", none); code != 200 || !strings.Contains(body, "cavp_engine_ready 0") {
		t.Errorf("/metrics %d: %s", code, body)
	}
}

// The readiness probe follows the engine's /v1/models: ready only while it lists a model.
func TestReadinessProbe(t *testing.T) {
	var models atomic.Value
	models.Store(`{"data":[]}`)
	eng := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, models.Load().(string)) })
	srv := newTestServer(t, eng)
	rd := newReadiness(srv.URL, 10*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go rd.Run(ctx)
	time.Sleep(50 * time.Millisecond)
	if rd.Ready() {
		t.Fatal("ready with an empty model list")
	}
	models.Store(`{"data":[{"id":"m"}]}`)
	waitFor(t, "ready", rd.Ready)
	srv.Close()
	waitFor(t, "not ready after the engine went away", func() bool { return !rd.Ready() })
}

// /v1/models is the engine's own list, unchanged.
func TestModelsPassThrough(t *testing.T) {
	h := newHarness(t, nil, nil)
	code, body := h.do("GET", "/v1/models", "", nil)
	if code != 200 || body != `{"object":"list","data":[{"id":"served-model","max_model_len":1000}]}` {
		t.Fatalf("%d %s", code, body)
	}
}

// /v1/tokenize reaches the engine's /tokenize with the body as sent, without waiting for
// admission.
func TestTokenizeForwardedUnadmitted(t *testing.T) {
	var seen atomic.Value
	h := newHarnessEngine(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen.Store(r.URL.Path + " " + string(b))
		w.Write([]byte(`{"tokens":[9]}`))
	}), func(c *Config) { c.Admission.MaxActive = 1 })
	rel, _, _ := h.p.admission.Acquire(context.Background(), AdmissionRequest{})
	defer rel()
	for _, path := range []string{"/v1/tokenize", "/tokenize"} {
		code, body := h.do("POST", path, `{"model":"m","prompt":"x"}`, nil)
		if code != 200 || body != `{"tokens":[9]}` {
			t.Fatalf("%s: %d %s", path, code, body)
		}
		if got := seen.Load(); got != `/tokenize {"model":"m","prompt":"x"}` {
			t.Fatalf("%s: engine saw %v", path, got)
		}
	}
}

// Credentials, identity headers and hop-by-hop headers stay at the proxy, on both
// forwarding paths (cv-56 #11, cv-54 #11).
func TestOnlyAllowlistedHeadersReachEngine(t *testing.T) {
	var seen atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Clone())
		w.Header().Set("X-Engine", "1")
		w.Write([]byte("{}"))
	}, func(c *Config) {
		c.Auth = auth.Config{Mode: "trusted_header", Secret: "0123456789abcdef-secret"}
	})
	hdr := map[string]string{
		"Authorization": "Bearer caller", "Proxy-Authorization": "Basic x", "Cookie": "a=b",
		auth.HeaderSecret: "0123456789abcdef-secret", auth.HeaderUser: "alice", auth.HeaderPriority: "3",
		"X-Request-Id": "req-1", "Keep-Alive": "timeout=1", "X-Priority": "-5", "X-Session-Key": "s",
	}
	for _, body := range []string{chatBody, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`} {
		resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", body, hdr)
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		got := seen.Load().(http.Header)
		for k := range got {
			switch k {
			case "Content-Type", "Content-Length", "User-Agent", "X-Request-Id", "Accept-Encoding", "Accept":
			default:
				t.Errorf("%s reached the engine", k)
			}
		}
		if got.Get("X-Request-Id") != "req-1" {
			t.Error("X-Request-Id not forwarded")
		}
	}
}

func TestBodyOverLimitIs413(t *testing.T) {
	var hits atomic.Int32
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }, func(c *Config) { c.MaxBodyBytes = 100 })
	big := `{"model":"m","messages":[{"role":"user","content":"` + strings.Repeat("x", 500) + `"}]}`
	if code, body := h.do("POST", "/v1/chat/completions", big, nil); code != 413 || !strings.Contains(body, "max_body_bytes") {
		t.Fatalf("declared length: %d %s", code, body)
	}
	if ev := h.event(1); num(ev["status"]) != 413 || num(ev["content_length"]) != len(big) {
		t.Fatalf("log: %v", ev)
	}
	// No declared length: refused once the limit is crossed, not truncated and parsed.
	req, _ := http.NewRequest("POST", h.srv.URL+"/v1/chat/completions", io.MultiReader(strings.NewReader(big)))
	req.Header.Set("Authorization", "Bearer "+testKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 413 {
		t.Fatalf("chunked: %d", resp.StatusCode)
	}
	if code, _ := h.do("POST", "/v1/tokenize", big, nil); code != 413 {
		t.Fatalf("tokenize: %d", code)
	}
	if hits.Load() != 0 {
		t.Fatal("an oversized body reached the engine")
	}
}

// A body cut short is a 400 that says so, with the declared and received sizes logged
// (cv-59 #10).
func TestTruncatedUploadIs400(t *testing.T) {
	h := newHarness(t, nil, nil)
	conn, err := net.Dial("tcp", h.srv.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %s\r\nContent-Length: 255\r\n\r\n%s", testKey, strings.Repeat("a", 60))
	conn.(*net.TCPConn).CloseWrite()
	ev := h.event(1)
	if num(ev["status"]) != 400 || num(ev["content_length"]) != 255 || num(ev["request_bytes"]) != 60 ||
		!strings.Contains(ev["error"].(string), "read request body") {
		t.Fatalf("log: %v", ev)
	}
}

func TestInvalidJSONIs400(t *testing.T) {
	h := newHarness(t, nil, nil)
	for _, body := range []string{"{", "[1]", "null", `"x"`} {
		if code, _ := h.do("POST", "/v1/chat/completions", body, nil); code != 400 {
			t.Errorf("%q: %d", body, code)
		}
	}
}

// The body is forwarded as sent apart from priority (and include_usage on streams): the
// model name passes through, and numbers keep their exact text.
func TestBodyPassesThrough(t *testing.T) {
	var seen atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen.Store(string(b))
		w.Write([]byte("{}"))
	}, nil)
	h.do("POST", "/v1/chat/completions", `{"model":"default","seed":12345678901234567890,"temperature":0.70,"messages":[{"role":"developer","content":"x"}]}`, nil)
	got := seen.Load().(string)
	for _, want := range []string{`"model":"default"`, `"seed":12345678901234567890`, `"temperature":0.70`, `"role":"developer"`} {
		if !strings.Contains(got, want) {
			t.Errorf("engine body %s lacks %s", got, want)
		}
	}
}

// The caller's identity is the most urgent it may run; the body's priority may only make
// it run later, clamped. X-Priority is not a control.
func TestPriorityFromBodyOnlyLater(t *testing.T) {
	var seen atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		seen.Store(b)
		w.Write([]byte("{}"))
	}, func(c *Config) {
		c.Auth.Keys = append(c.Auth.Keys, auth.StaticKey{Key: "k5", User: "bob", Priority: 5})
	})
	for _, c := range []struct {
		key, prio string
		want      any
	}{
		{testKey, "", nil}, {testKey, `,"priority":7`, 7.0}, {testKey, `,"priority":-3`, nil},
		{"k5", "", 5.0}, {"k5", `,"priority":-3`, 5.0}, {"k5", `,"priority":500`, 100.0},
		{"k5", `,"priority":7.5`, 5.0}, {"k5", `,"priority":"9"`, 5.0},
	} {
		body := `{"model":"m","messages":[{"role":"user","content":"hi"}]` + c.prio + `}`
		h.do("POST", "/v1/chat/completions", body, map[string]string{"Authorization": "Bearer " + c.key, "X-Priority": "-10"})
		if got := seen.Load().(map[string]any)["priority"]; got != c.want {
			t.Errorf("key %s body %q: engine priority %v, want %v", c.key, c.prio, got, c.want)
		}
	}
}

// Sessions come from the conversation and the caller only; X-Session-Key is ignored.
func TestSessionFromConversation(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("{}")) }, func(c *Config) {
		c.Auth.Keys = append(c.Auth.Keys, auth.StaticKey{Key: "k2", User: "bob"})
	})
	h.do("POST", "/v1/chat/completions", chatBody, map[string]string{"X-Session-Key": "a"})
	h.do("POST", "/v1/chat/completions", chatBody, map[string]string{"X-Session-Key": "b"})
	h.do("POST", "/v1/chat/completions", chatBody, map[string]string{"Authorization": "Bearer k2"})
	s1, s2, s3 := h.event(1)["session"], h.event(2)["session"], h.event(3)["session"]
	if s1 == nil || s1 != s2 {
		t.Errorf("same conversation, different sessions: %v %v", s1, s2)
	}
	if s1 == s3 {
		t.Error("two callers share a session")
	}
}

// Usage is read from a real vLLM body, where fields follow usage (cv-59 #1).
func TestNonStreamingUsage(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, realVLLMChatResponse)
	}, nil)
	code, body := h.do("POST", "/v1/chat/completions", chatBody, nil)
	if code != 200 || body != realVLLMChatResponse {
		t.Fatalf("%d %s", code, body)
	}
	ev := h.event(1)
	if num(ev["prompt_tokens"]) != 78 || num(ev["cached_prompt_tokens"]) != 64 || ev["usage_details"] != true ||
		num(ev["completion_tokens"]) != 5 || num(ev["status"]) != 200 || ev["error"] != nil {
		t.Fatalf("log: %v", ev)
	}
	_, m := h.do("GET", "/metrics", "", nil)
	for _, want := range []string{`cavp_requests_total{route="chat",status="200"} 1`, "cavp_prompt_tokens_total 78", "cavp_cached_prompt_tokens_total 64"} {
		if !strings.Contains(m, want) {
			t.Errorf("metrics lack %q:\n%s", want, m)
		}
	}
}

// A response larger than the usage copy is still forwarded whole.
func TestLargeResponseForwardedWhole(t *testing.T) {
	big := `{"x":"` + strings.Repeat("a", maxUsageBodyBytes) + `","usage":{"prompt_tokens":1,"total_tokens":2}}`
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, big) }, nil)
	code, body := h.do("POST", "/v1/chat/completions", chatBody, nil)
	if code != 200 || body != big {
		t.Fatalf("%d, %d bytes", code, len(body))
	}
	if ev := h.event(1); ev["prompt_tokens"] != nil {
		t.Fatalf("usage from a body that was not kept: %v", ev)
	}
}

// While the engine is not ready, requests wait instead of failing, and go once it is.
func TestRequestsWaitForReadiness(t *testing.T) {
	var hits atomic.Int32
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write([]byte(realVLLMChatResponse))
	}, nil)
	h.p.ready.set(false, "test")
	done := make(chan int)
	go func() {
		code, _ := h.do("POST", "/v1/chat/completions", chatBody, nil)
		done <- code
	}()
	time.Sleep(200 * time.Millisecond)
	if hits.Load() != 0 {
		t.Fatal("forwarded while the engine was not ready")
	}
	h.p.ready.set(true, "")
	if code := <-done; code != 200 {
		t.Fatalf("status %d", code)
	}
	if ev := h.event(1); num(ev["ready_wait_ms"]) < 150 {
		t.Fatalf("log: %v", ev)
	}
}

// A request whose client leaves while it is queued gets a log line (cv-54 #13, cv-59 #3).
func TestAbandonedWhileQueuedIsLogged(t *testing.T) {
	h := newHarness(t, nil, func(c *Config) { c.Admission.MaxActive = 1 })
	rel, _, _ := h.p.admission.Acquire(context.Background(), AdmissionRequest{})
	defer rel()
	for i, body := range []string{chatBody, `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`} {
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			waitFor(t, "queued", func() bool { return h.p.admission.Snapshot().Queued == 1 })
			cancel()
		}()
		if resp, err := h.request(ctx, "POST", "/v1/chat/completions", body, nil); err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
		ev := h.event(i + 1)
		if num(ev["status"]) != statusClientGone || ev["error"] == nil {
			t.Fatalf("log: %v", ev)
		}
	}
}

// Shutting down answers waiting requests 503 rather than cutting them silently (cv-53 #4).
func TestShutdownAnswersWaiters(t *testing.T) {
	h := newHarness(t, nil, nil)
	h.p.ready.set(false, "test")
	done := make(chan int)
	go func() {
		code, _ := h.do("POST", "/v1/chat/completions", chatBody, nil)
		done <- code
	}()
	time.Sleep(100 * time.Millisecond)
	h.p.Close()
	if code := <-done; code != 503 {
		t.Fatalf("status %d", code)
	}
}

// An unreachable engine is a 502 that names no internal address.
func TestEngineFailureIs502WithoutDetail(t *testing.T) {
	h := newHarness(t, nil, nil)
	h.eng.Close()
	code, body := h.do("POST", "/v1/chat/completions", chatBody, nil)
	if code != 502 || strings.Contains(body, "127.0.0.1") {
		t.Fatalf("%d %s", code, body)
	}
	if ev := h.event(1); !strings.Contains(fmt.Sprint(ev["error"]), "engine request failed") {
		t.Fatalf("log: %v", ev)
	}
}

// A client that gives up while the engine works is logged as gone, not as an engine
// failure (cv-55 #11).
func TestClientCancelIsNotAnEngineFailure(t *testing.T) {
	started := make(chan struct{})
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body) // so the server notices the disconnect
		close(started)
		<-r.Context().Done()
	}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; cancel() }()
	h.request(ctx, "POST", "/v1/chat/completions", chatBody, nil)
	if ev := h.event(1); num(ev["status"]) != statusClientGone {
		t.Fatalf("log: %v", ev)
	}
}

// An engine response cut mid-body aborts the client's connection instead of ending the
// response as if it were complete.
func TestEngineResponseCutAbortsClient(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"partial":`))
		w.(http.Flusher).Flush()
		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}, nil)
	resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", chatBody, nil)
	if err == nil {
		_, err = io.ReadAll(resp.Body)
		resp.Body.Close()
	}
	if err == nil {
		t.Fatal("the client saw a complete response")
	}
	if ev := h.event(1); !strings.Contains(fmt.Sprint(ev["error"]), "engine response cut") {
		t.Fatalf("log: %v", ev)
	}
}

// A client that stops reading (without disconnecting) loses its admission permit once a
// write stalls past the guard, on both paths (cv-58 #3, cv-53 #7).
func TestStalledClientReleasesPermit(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint("stream=", stream), func(t *testing.T) {
			h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
				chunk := []byte("data: " + strings.Repeat("x", 64<<10) + "\n\n")
				for r.Context().Err() == nil {
					if _, err := w.Write(chunk); err != nil {
						return
					}
				}
			}, nil)
			h.p.writeStall = 200 * time.Millisecond
			conn, err := net.Dial("tcp", h.srv.Listener.Addr().String())
			if err != nil {
				t.Fatal(err)
			}
			defer conn.Close()
			body := fmt.Sprintf(`{"model":"m","stream":%v,"messages":[]}`, stream)
			fmt.Fprintf(conn, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer %s\r\nContent-Length: %d\r\n\r\n%s", testKey, len(body), body)
			// Read nothing.
			waitFor(t, "admitted", func() bool { return h.p.admission.Snapshot().Active == 1 })
			waitFor(t, "permit released", func() bool { return h.p.admission.Snapshot().Active == 0 })
			if ev := h.event(1); !strings.Contains(fmt.Sprint(ev["error"]), "client went away") {
				t.Fatalf("log: %v", ev)
			}
		})
	}
}

func TestMetrics(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("{}")) }, nil)
	h.do("POST", "/v1/chat/completions", chatBody, nil)
	h.do("POST", "/v1/nothing", chatBody, nil)
	h.event(2)
	_, body := h.do("GET", "/metrics", "", map[string]string{"Authorization": ""})
	for _, want := range []string{
		"# TYPE cavp_queue_length gauge\ncavp_queue_length 0\n",
		"cavp_active_requests 0\n", "cavp_cache_used_tokens ", "cavp_cache_budget_tokens ",
		"cavp_cache_held_tokens ", "cavp_guard_margin_blocks 0\n", "# TYPE cavp_guard_raises_total counter\n",
		"# TYPE cavp_cache_displaced_tokens_total counter\ncavp_cache_displaced_tokens_total 0\n", "cavp_cache_holds_evicted_total 0\n",
		"cavp_kv_mirror_connected 0\n", "cavp_engine_ready 1\n",
		`cavp_requests_total{route="chat",status="200"} 1`, `cavp_requests_total{route="none",status="404"} 1`,
		"cavp_prompt_tokens_total 0\n", "cavp_cached_prompt_tokens_total 0\n",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics lack %q", want)
		}
	}
	// Every sample line parses as name{labels} value.
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		if f := strings.Fields(line); len(f) != 2 || !strings.HasPrefix(f[0], "cavp_") {
			t.Errorf("bad sample line %q", line)
		}
	}
}

// The write guard's deadline does not outlive its request on a kept-alive connection.
func TestWriteDeadlineClearedForNextRequest(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("{}")) }, nil)
	h.p.writeStall = 50 * time.Millisecond
	tr := &http.Transport{MaxConnsPerHost: 1}
	defer tr.CloseIdleConnections()
	c := &http.Client{Transport: tr}
	for i := 0; i < 2; i++ {
		req, _ := http.NewRequest("POST", h.srv.URL+"/v1/chat/completions", strings.NewReader(chatBody))
		req.Header.Set("Authorization", "Bearer "+testKey)
		resp, err := c.Do(req)
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		time.Sleep(150 * time.Millisecond)
	}
}

// Until the engine's KV pool is known nothing is admitted: requests wait, then run.
func TestRequestsWaitForThePool(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(realVLLMChatResponse)) }, nil)
	h.p.admission.SetPool(0, 16)
	done := make(chan int)
	go func() {
		code, _ := h.do("POST", "/v1/chat/completions", chatBody, nil)
		done <- code
	}()
	waitFor(t, "queued", func() bool { return h.p.admission.Snapshot().Queued == 1 })
	h.p.admission.SetPool(1000, 16)
	if code := <-done; code != 200 {
		t.Fatalf("status %d", code)
	}
}

// A body must be exactly one JSON object: trailing data is refused, not dropped
// (rev-1 #3).
func TestTrailingDataIs400(t *testing.T) {
	var hits atomic.Int32
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }, nil)
	for _, body := range []string{chatBody + " garbage", chatBody + chatBody, chatBody + " {}"} {
		if code, _ := h.do("POST", "/v1/chat/completions", body, nil); code != 400 {
			t.Errorf("%q: %d", body, code)
		}
	}
	if code, _ := h.do("POST", "/v1/chat/completions", chatBody+" \n\t", nil); code != 200 {
		t.Errorf("trailing whitespace: %d", code)
	}
	if hits.Load() != 1 {
		t.Fatalf("engine saw %d requests", hits.Load())
	}
}

// The hit-ratio counters count only requests that reported cache details: without them
// a hit is unknown, not a miss (rev-1 #4).
func TestHitRatioCountsOnlyReportedDetails(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"usage":{"prompt_tokens":100,"completion_tokens":1,"total_tokens":101}}`)
	}, nil)
	h.do("POST", "/v1/chat/completions", chatBody, nil)
	h.event(1)
	_, m := h.do("GET", "/metrics", "", nil)
	if !strings.Contains(m, "cavp_prompt_tokens_total 0\n") {
		t.Fatalf("prompt tokens without details counted:\n%s", m)
	}
}
