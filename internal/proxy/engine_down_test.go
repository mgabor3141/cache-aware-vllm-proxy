package proxy

import (
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// A running request on an engine that went silent and fails its readiness probe is
// given up, so its permit is released (chaos #8, vl-75). Streaming and not.
func TestEngineDownReleasesSilentRequest(t *testing.T) {
	for _, body := range []string{streamBody, chatBody} {
		in := make(chan struct{}, 1)
		h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
			io.Copy(io.Discard, r.Body)
			in <- struct{}{}
			<-r.Context().Done() // a hung engine: accepts, never answers
		}, nil)
		h.p.engineDownAfter = 100 * time.Millisecond
		done := make(chan string)
		go func() {
			_, b := h.do("POST", "/v1/chat/completions", body, nil)
			done <- b
		}()
		<-in
		h.p.ready.set(false, "test")
		waitFor(t, "permit released", func() bool { return h.p.admission.Snapshot().Active == 0 })
		b := <-done
		ev := h.event(1)
		if num(ev["status"]) != 502 || !strings.Contains(ev["error"].(string), "readiness") {
			t.Fatalf("log %v", ev)
		}
		if strings.Contains(body, `"stream":true`) && !strings.Contains(b, `"error"`) {
			t.Fatalf("stream ended without an error event: %q", b)
		}
	}
}

// A stream that keeps receiving from the engine is not cut while the probe fails.
func TestEngineDownKeepsStreamThatIsHeard(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		sseEngine(sseChunk)(w, r)
		for i := 0; i < 20; i++ {
			time.Sleep(20 * time.Millisecond)
			sseEngine(sseChunk)(w, r)
		}
		sseEngine(sseDone)(w, r)
	}, nil)
	h.p.engineDownAfter = 100 * time.Millisecond
	done := make(chan string)
	go func() {
		_, b := h.do("POST", "/v1/chat/completions", streamBody, nil)
		done <- b
	}()
	waitFor(t, "streaming", func() bool { return h.p.admission.Snapshot().Active == 1 })
	time.Sleep(50 * time.Millisecond) // past the admission check, chunks flowing
	h.p.ready.set(false, "test")
	b := <-done
	if !strings.HasSuffix(b, sseDone) || strings.Contains(b, `"error"`) {
		t.Fatalf("%q", b)
	}
	if ev := h.event(1); num(ev["status"]) != 200 {
		t.Fatalf("log %v", ev)
	}
}
