package proxy

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// In raw mode a keepalive is never written inside an event the engine left open, and is
// written when the stream is silent at an event boundary.
func TestRawModeKeepalivesFollowEventBoundary(t *testing.T) {
	for _, c := range []struct {
		chunk string
		open  bool
	}{
		{"data: {\"choices\":[{\"de", true},
		{"data: x\n\n", false},
	} {
		rec := &recorder{}
		sw := &sseWriter{w: rec, cancel: func() {}, raw: true}
		if err := sw.relay([]byte(c.chunk)); err != nil {
			t.Fatal(err)
		}
		if sw.open != c.open {
			t.Fatalf("%q: open=%v", c.chunk, sw.open)
		}
		stop := sw.keepalive(20 * time.Millisecond)
		time.Sleep(150 * time.Millisecond)
		stop()
		if got := strings.Contains(rec.String(), "keepalive"); got == c.open {
			t.Errorf("%q: keepalive written=%v: %q", c.chunk, got, rec.String())
		}
	}
}

// A stream still waiting when the proxy shuts down gets a 503 error event and [DONE].
func TestShutdownAnswersAWaitingStream(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("the engine saw a request that was still waiting")
	}, nil)
	h.p.keepalive = 30 * time.Millisecond
	h.p.ready.set(false, "test")
	resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", streamBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	h.p.Close()
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatalf("the stream did not end: %v", err)
	}
	if !strings.Contains(string(body), `"code":503`) || !strings.HasSuffix(string(body), sseDone) {
		t.Fatalf("body %q", body)
	}
	if ev := h.event(1); num(ev["status"]) != 503 {
		t.Fatalf("log %v", ev)
	}
}

// An engine cutting a buffered response returns the permit.
func TestEngineCutReturnsThePermit(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"partial":`))
		w.(http.Flusher).Flush()
		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}, nil)
	if resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", chatBody, nil); err == nil {
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	waitFor(t, "the permit back", func() bool { return h.p.admission.Snapshot().Active == 0 })
}

// A waiter granted at the moment its client gives up hands the permit straight back.
func TestReleaseVersusCancelReturnsThePermit(t *testing.T) {
	branch := 0
	for i := 0; i < 2000; i++ {
		q := slotQueue(1)
		hold, _, ok := q.Acquire(context.Background(), AdmissionRequest{Session: "h"})
		if !ok {
			t.Fatal("setup: the first permit")
		}
		ctx, cancel := context.WithCancel(context.Background())
		admitted := make(chan func(), 1)
		go func() {
			rel, _, ok := q.Acquire(ctx, AdmissionRequest{Session: "s"})
			if !ok {
				rel = nil
			}
			admitted <- rel
		}()
		waitForQueued(t, q, 1)
		go cancel()
		hold()
		rel := <-admitted
		if rel != nil {
			rel()
		}
		waitFor(t, "the queue empty", func() bool {
			q.mu.Lock()
			defer q.mu.Unlock()
			return q.sessions["s"].waiting == 0 && len(q.running) == 0 && q.active == 0
		})
		q.mu.Lock()
		if rel == nil && q.sessions["s"].gaveUpSince.IsZero() {
			branch++ // granted, then cancelled
		}
		q.mu.Unlock()
	}
	if branch == 0 {
		t.Fatal("never reached the granted-then-cancelled branch")
	}
}
