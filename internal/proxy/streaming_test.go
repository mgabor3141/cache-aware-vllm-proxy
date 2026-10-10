// streaming_test.go — verify the keepalive-aware streaming handler.
//
// We're testing behaviour the user actually cares about, not internal
// state:
//
//  1. While upstream is unavailable, the client receives SSE keepalive
//     lines so its connection doesn't time out.
//  2. Once upstream becomes available, real SSE events flow through.
//  3. If the engine stays down past the recovery timeout, the client receives an SSE error event
//     and a [DONE] marker so the chat UI can render a clean error
//     rather than hanging.
//  4. The Authorization header is stripped before reaching upstream.
package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
)

// makeTestProxy builds a minimal Proxy with the given upstream URL
// and retry config, wired with a real *httputil.ReverseProxy (its
// .Transport field is what serveStreamingWithKeepalive reads). No
// auth, no resolver: those paths aren't exercised by streaming
// tests.
func makeTestProxy(t *testing.T, upstreamURL string, ready *atomic.Bool) *Proxy {
	t.Helper()
	u, err := url.Parse(upstreamURL)
	if err != nil {
		t.Fatalf("parse upstream: %v", err)
	}
	h := liveHealth(t, ready)
	flakey := &flakeyTransport{base: http.DefaultTransport}
	rp := &httputil.ReverseProxy{
		Transport: newHealthAwareRetryingTransport(flakey, h.cfg, h),
	}
	return &Proxy{
		cfg:      &Config{Upstream: upstreamURL, Retry: h.cfg},
		upstream: u,
		logger:   NewEventLogger(io.Discard),
		rp:       rp,
		health:   h,
	}
}

// flakeyOf reaches into the proxy's retrying transport stack and
// returns the embedded *flakeyTransport so tests can configure
// its per-test failure count and read its attempt counter.
func flakeyOf(p *Proxy) *flakeyTransport {
	rt := p.rp.Transport.(*retryingTransport)
	return rt.base.(*flakeyTransport)
}

func TestStreaming_Success_AfterTransientUpstreamFailures(t *testing.T) {
	// Upstream that refuses connections for the first 2 attempts,
	// then accepts and returns a single SSE chunk + [DONE]. We
	// verify the client sees the real chunk after the retries
	// complete and that the auth header is stripped on the way
	// upstream.
	realUp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Errorf("upstream saw Authorization header: %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"hi"}}]}`+"\n\n")
		flusher.Flush()
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer realUp.Close()

	ready := &atomic.Bool{}
	ready.Store(true)
	p := makeTestProxy(t, realUp.URL, ready)
	flakey := flakeyOf(p)
	flakey.failures = 2

	req := buildStreamReq(t)
	rec := httptest.NewRecorder()
	ev := &Event{Path: "/v1/chat/completions"}
	p.serveStreamingWithKeepalive(rec, req, ev)

	if rec.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected 200 (streaming committed), got %d", rec.Result().StatusCode)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `data: {"choices":[{"delta":{"content":"hi"}}]}`) {
		t.Fatalf("missing real SSE chunk; got body:\n%s", body)
	}
	if !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("missing [DONE] marker; got body:\n%s", body)
	}
	if got := flakey.attempts.Load(); got != 3 {
		t.Fatalf("expected 3 upstream attempts (2 fail + 1 success), got %d", got)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("expected Content-Type text/event-stream, got %q", ct)
	}
}

func TestStreaming_RetriesExhausted_EmitsSSEError(t *testing.T) {
	// Upstream never recovers within the retry budget. We've
	// already committed HTTP 200 (sent SSE headers immediately so
	// the client connection stays alive), so the failure has to
	// surface as an SSE error event + [DONE] marker. Chat UIs
	// render this as an in-conversation error message rather than
	// a transport failure.
	ready := &atomic.Bool{}
	ready.Store(true)
	p := makeTestProxy(t, "http://placeholder", ready)
	flakey := flakeyOf(p)
	flakey.failures = 100                         // always fail
	flakey.onFail = func() { ready.Store(false) } // and the engine stays down

	req := buildStreamReq(t)
	rec := httptest.NewRecorder()
	ev := &Event{Path: "/v1/chat/completions"}
	p.serveStreamingWithKeepalive(rec, req, ev)

	if rec.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected 200 (streaming committed), got %d", rec.Result().StatusCode)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"error"`) {
		t.Fatalf("missing error event in SSE; got body:\n%s", body)
	}
	if !strings.Contains(body, `"proxy_retry_exhausted"`) {
		t.Fatalf("missing proxy_retry_exhausted code; got body:\n%s", body)
	}
	if !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("missing [DONE] marker after error; got body:\n%s", body)
	}
	// Verify the SSE error payload parses as JSON.
	found := false
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, `data: {"error"`) {
			payload := strings.TrimPrefix(line, "data: ")
			var v map[string]any
			if err := json.Unmarshal([]byte(payload), &v); err != nil {
				t.Fatalf("error payload is not valid JSON: %v\n%s", err, payload)
			}
			found = true
			break
		}
	}
	if !found {
		t.Fatal("didn't find an SSE data line containing an error JSON")
	}
}

// flakeyTransport fails the first N attempts with connection-refused
// errors, then forwards subsequent attempts to the embedded base
// transport. Used to simulate a vLLM restart that lasts N attempts.
type flakeyTransport struct {
	attempts atomic.Int32
	failures int
	onFail   func()
	base     http.RoundTripper
}

func (f *flakeyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	n := f.attempts.Add(1)
	if int(n) <= f.failures {
		if f.onFail != nil {
			f.onFail()
		}
		return nil, fmt.Errorf("dial tcp 10.0.0.2:8000: connect: connection refused")
	}
	return f.base.RoundTrip(req)
}

// buildStreamReq builds a POST /v1/chat/completions request with a
// stream:true body and an Authorization header (which the streaming
// handler should strip before sending upstream). Sets GetBody so
// the retrying transport can replay.
func buildStreamReq(t *testing.T) *http.Request {
	t.Helper()
	body := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req, err := http.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("new req: %v", err)
	}
	req.Header.Set("Authorization", "Bearer test")
	req.Header.Set("Content-Type", "application/json")
	req.ContentLength = int64(len(body))
	buf := body
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(buf)), nil
	}
	return req
}
