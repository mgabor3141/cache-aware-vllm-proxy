package proxy

import (
	"context"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
)

// Every API request has an id: the caller's X-Request-Id, else a new one. It is logged,
// echoed to the client and sent to the engine (operator #27).
func TestRequestIDLoggedEchoedForwarded(t *testing.T) {
	var seen atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		seen.Store(r.Header.Get("X-Request-Id"))
		w.Write([]byte(realVLLMChatResponse))
	}, nil)
	for i, given := range []string{"req-abc", ""} {
		resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", chatBody, map[string]string{"X-Request-Id": given})
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		echoed := resp.Header.Get("X-Request-Id")
		logged, _ := h.event(i + 1)["request_id"].(string)
		if echoed == "" || logged != echoed || seen.Load() != echoed || (given != "" && echoed != given) {
			t.Fatalf("given %q: echoed %q logged %q engine %q", given, echoed, logged, seen.Load())
		}
	}
	// Refusals carry it too.
	resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", chatBody, map[string]string{"Authorization": "Bearer nope"})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 || resp.Header.Get("X-Request-Id") == "" {
		t.Fatalf("401 without a request id: %d %v", resp.StatusCode, resp.Header)
	}
}
