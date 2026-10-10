package proxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

type mockTransport struct {
	calls atomic.Int32
	fn    func(attempt int, req *http.Request) (*http.Response, error)
}

func (m *mockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	n := m.calls.Add(1)
	return m.fn(int(n), req)
}

func mkResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(bytes.NewBufferString(body)),
		Header:     http.Header{},
	}
}

// liveHealth runs an upstreamHealth whose readiness probe answers while ready is true,
// and returns once it reports healthy.
func liveHealth(t *testing.T, ready *atomic.Bool, mutate ...func(*RetryConfig)) *upstreamHealth {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && ready.Load() {
			w.Write([]byte(`{"data":[{"id":"m"}]}`))
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)
	cfg := testHealthConfig()
	cfg.RetryStatusCodes = []int{502, 503, 504}
	for _, m := range mutate {
		m(&cfg)
	}
	h := newUpstreamHealth(srv.URL, cfg, srv.Client())
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go h.Run(ctx)
	deadline := time.Now().Add(2 * time.Second)
	for !h.Ready() {
		if time.Now().After(deadline) {
			t.Fatal("health never became ready")
		}
		time.Sleep(time.Millisecond)
	}
	return h
}

func retryRig(t *testing.T, fn func(int, *http.Request) (*http.Response, error), mutate ...func(*RetryConfig)) (*retryingTransport, *mockTransport, *atomic.Bool) {
	t.Helper()
	ready := &atomic.Bool{}
	ready.Store(true)
	h := liveHealth(t, ready, mutate...)
	m := &mockTransport{fn: fn}
	return newHealthAwareRetryingTransport(m, h.cfg, h), m, ready
}

func get(t *testing.T, rt http.RoundTripper) (*http.Response, error) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, "http://upstream/v1/models", nil)
	return rt.RoundTrip(req)
}

func TestRetry_TransientNetworkError_RetriedAfterRecovery(t *testing.T) {
	rt, m, _ := retryRig(t, func(n int, _ *http.Request) (*http.Response, error) {
		if n == 1 {
			return nil, errors.New("dial tcp: connect: connection refused")
		}
		return mkResponse(200, "ok"), nil
	})
	resp, err := get(t, rt)
	if err != nil || resp.StatusCode != 200 || m.calls.Load() != 2 {
		t.Fatalf("resp=%v err=%v calls=%d", resp, err, m.calls.Load())
	}
}

func TestRetry_RetryableStatus_RetriedAfterRecovery(t *testing.T) {
	rt, m, _ := retryRig(t, func(n int, _ *http.Request) (*http.Response, error) {
		if n == 1 {
			return mkResponse(503, "starting"), nil
		}
		return mkResponse(200, "ok"), nil
	})
	resp, err := get(t, rt)
	if err != nil || resp.StatusCode != 200 || m.calls.Load() != 2 {
		t.Fatalf("resp=%v err=%v calls=%d", resp, err, m.calls.Load())
	}
}

func TestRetry_4xxNotRetried(t *testing.T) {
	rt, m, _ := retryRig(t, func(int, *http.Request) (*http.Response, error) { return mkResponse(404, ""), nil })
	resp, err := get(t, rt)
	if err != nil || resp.StatusCode != 404 || m.calls.Load() != 1 {
		t.Fatalf("resp=%v err=%v calls=%d", resp, err, m.calls.Load())
	}
}

func TestRetry_NonRetryableNetworkError_NotRetried(t *testing.T) {
	rt, m, _ := retryRig(t, func(int, *http.Request) (*http.Response, error) {
		return nil, errors.New("dial tcp: lookup upstream: no such host")
	})
	if _, err := get(t, rt); err == nil || m.calls.Load() != 1 {
		t.Fatalf("err=%v calls=%d", err, m.calls.Load())
	}
}

// When the engine stays down past recovery_timeout, the request fails as unavailable.
func TestRetry_GivesUpAfterRecoveryTimeout(t *testing.T) {
	var ready *atomic.Bool
	rt, _, ready := retryRig(t, func(int, *http.Request) (*http.Response, error) {
		ready.Store(false) // the engine went away with the request
		return nil, errors.New("connection refused")
	})
	start := time.Now()
	_, err := get(t, rt)
	if !errors.Is(err, ErrUpstreamUnavailable) {
		t.Fatalf("err=%v, want ErrUpstreamUnavailable", err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatalf("took %s", time.Since(start))
	}
}

func TestRetry_ContextCancelWhileWaiting_ReturnsPromptly(t *testing.T) {
	var ready *atomic.Bool
	rt, _, ready := retryRig(t, func(int, *http.Request) (*http.Response, error) {
		ready.Store(false)
		return nil, errors.New("connection refused")
	}, func(c *RetryConfig) { c.RecoveryTimeout = time.Hour })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "http://upstream/x", nil)
	start := time.Now()
	if _, err := rt.RoundTrip(req); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err=%v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("took %s", time.Since(start))
	}
}

func TestRetry_POSTBodyReplayedViaGetBody(t *testing.T) {
	var seen []string
	rt, _, _ := retryRig(t, func(n int, req *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(req.Body)
		seen = append(seen, string(b))
		if n == 1 {
			return nil, errors.New("connection reset by peer")
		}
		return mkResponse(200, ""), nil
	})
	body := []byte(`{"x":1}`)
	req, _ := http.NewRequest(http.MethodPost, "http://upstream/v1/chat/completions", bytes.NewReader(body))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	if _, err := rt.RoundTrip(req); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0] != string(body) || seen[1] != string(body) {
		t.Fatalf("bodies %q", seen)
	}
}

func TestRetry_POSTBodyWithoutGetBody_FailsOnRetry(t *testing.T) {
	rt, m, _ := retryRig(t, func(int, *http.Request) (*http.Response, error) {
		return nil, errors.New("connection refused")
	})
	req, _ := http.NewRequest(http.MethodPost, "http://upstream/x", bytes.NewReader([]byte("x")))
	req.GetBody = nil
	_, err := rt.RoundTrip(req)
	if err == nil || m.calls.Load() != 1 {
		t.Fatalf("err=%v calls=%d", err, m.calls.Load())
	}
	if want := "no GetBody"; !bytes.Contains([]byte(fmt.Sprint(err)), []byte(want)) {
		t.Fatalf("err=%v, want %q", err, want)
	}
}
