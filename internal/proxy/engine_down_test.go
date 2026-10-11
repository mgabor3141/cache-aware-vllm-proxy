package proxy

import (
	"context"
	"io"
	"net"
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
		h.p.ready.noteGone(true)
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
	h.p.ready.noteGone(true)
	b := <-done
	if !strings.HasSuffix(b, sseDone) || strings.Contains(b, `"error"`) {
		t.Fatalf("%q", b)
	}
	if ev := h.event(1); num(ev["status"]) != 200 {
		t.Fatalf("log %v", ev)
	}
}

// A probe that fails without showing the engine gone (a timeout: the engine is busy)
// never gives up a running request, even one the engine is silent on (rev-3 #1).
func TestEngineBusyProbeFailureKeepsRequest(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(600 * time.Millisecond)
		w.Write([]byte(realVLLMChatResponse))
	}, nil)
	h.p.engineDownAfter = 100 * time.Millisecond
	go func() {
		time.Sleep(50 * time.Millisecond)
		h.p.ready.set(false, "probe timed out")
		h.p.ready.noteGone(false)
	}()
	if code, b := h.do("POST", "/v1/chat/completions", chatBody, nil); code != 200 {
		t.Fatalf("%d %s", code, b)
	}
}

// Only errors that show nothing listening count as the engine gone.
func TestEngineGoneClassification(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	h := newReadiness("http://"+addr, time.Second)
	if err := h.probe(context.Background()); err == nil || !engineGone(err) {
		t.Fatalf("refused: %v gone=%v", err, err != nil && engineGone(err))
	}
	hang := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	h = newReadiness(hang.URL, time.Second)
	h.client.Timeout = 100 * time.Millisecond
	if err := h.probe(context.Background()); err == nil || engineGone(err) {
		t.Fatalf("timeout: %v counted as gone", err)
	}
	busy := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "loading", http.StatusServiceUnavailable)
	}))
	h = newReadiness(busy.URL, time.Second)
	if err := h.probe(context.Background()); err == nil || engineGone(err) {
		t.Fatalf("503: %v counted as gone", err)
	}
}
