package proxy

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// [DONE] ends the relay even when the engine keeps the response open: the client's
// stream ends, the permit comes back and the turn leaves its hold.
func TestDoneEndsStreamWithEngineResponseOpen(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, sseUsage+sseDone)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		case <-time.After(10 * time.Second):
		}
	}, nil)
	h.p.admission.cfg.MaxActive = 1
	done := make(chan string, 1)
	go func() {
		_, b := h.do("POST", "/v1/chat/completions", streamBody, nil)
		done <- b
	}()
	select {
	case b := <-done:
		if !strings.HasSuffix(b, sseDone) {
			t.Fatalf("body %q", b)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the client's stream did not end at [DONE]")
	}
	if ev := h.event(1); num(ev["status"]) != 200 {
		t.Fatalf("log %v", ev)
	}
	waitFor(t, "the permit back", func() bool { return h.p.admission.Snapshot().Active == 0 })
}
