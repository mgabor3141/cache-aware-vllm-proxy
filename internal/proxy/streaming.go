// Streaming requests (stream: true). The response is committed at once: SSE headers,
// then keepalive comments while the request waits for the engine and for admission,
// then the engine's stream as it arrives. Keepalives continue whenever the stream is
// silent, e.g. while a long prompt prefills.
//
// Once the 200 is out, failures reach the client as one SSE error event followed by
// "data: [DONE]"; the log line carries the real outcome (the engine's status, and an
// error when the stream failed or ended without [DONE]).
package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

const keepaliveLine = ": keepalive\n\n"

// maxErrorBodyBytes bounds how much of an engine error response is read.
const maxErrorBodyBytes = 64 << 10

func (p *Proxy) serveStream(w http.ResponseWriter, r *http.Request, ev *Event, body []byte) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no") // no buffering in nginx-style proxies in front
	w.WriteHeader(http.StatusOK)
	ev.Status = http.StatusOK

	// A failed write (client gone, or stalled past writeStallTimeout) cancels ctx, which
	// ends the wait or the upstream request.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	sw := &sseWriter{w: w, cancel: cancel}
	sw.flush()
	defer sw.keepalive(p.keepalive)()

	release, err := p.admit(ctx, r, ev)
	if err != nil {
		waitOutcome(ev, err)
		if ev.Status == http.StatusServiceUnavailable {
			sw.errorEvent("the proxy is shutting down; retry", http.StatusServiceUnavailable)
		}
		return
	}
	defer release() // runs after the usage below is recorded: admission reads it

	req, err := p.upstreamRequest(ctx, r, r.URL.Path, body)
	if err != nil {
		ev.Status, ev.Error = http.StatusInternalServerError, err.Error()
		sw.errorEvent("internal error", ev.Status)
		return
	}
	resp, err := p.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			ev.Status, ev.Error = statusClientGone, "client went away"
			return
		}
		ev.Status, ev.Error = http.StatusBadGateway, "engine request failed: "+err.Error()
		sw.errorEvent("inference upstream request failed", ev.Status)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// The engine refused the request (a 400 for a prompt over the context length, say).
		// Its message is meant for the client; pass it on, bounded.
		ev.Status = resp.StatusCode
		msg := engineErrorMessage(resp.Body)
		ev.Error = "engine: " + msg
		sw.errorEvent(msg, resp.StatusCode)
		return
	}

	scan := &usageScanner{chunks: ev.generated}
	defer func() {
		if scan.found {
			ev.setUsage(scan.usage)
		}
	}()
	buf := make([]byte, 32<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			scan.Write(buf[:n])
			if werr := sw.write(buf[:n]); werr != nil {
				ev.Error = "client went away mid-stream: " + werr.Error()
				return
			}
		}
		if rerr == io.EOF {
			if !scan.done {
				ev.Error = "stream ended without [DONE]"
				sw.errorEvent("the engine ended the stream early", http.StatusBadGateway)
			}
			return
		}
		if rerr != nil {
			if r.Context().Err() != nil {
				ev.Error = "client went away mid-stream"
				return
			}
			ev.Error = "engine stream cut: " + rerr.Error()
			sw.errorEvent("the engine stream failed", http.StatusBadGateway)
			return
		}
	}
}

// engineErrorMessage reads an engine error body (bounded) and returns its message:
// vLLM's {"message": ...} or OpenAI's {"error": {"message": ...}}, else the raw text.
func engineErrorMessage(r io.Reader) string {
	raw, _ := io.ReadAll(io.LimitReader(r, maxErrorBodyBytes))
	var e struct {
		Message string `json:"message"`
		Error   struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	msg := string(raw)
	if json.Unmarshal(raw, &e) == nil {
		if e.Message != "" {
			msg = e.Message
		} else if e.Error.Message != "" {
			msg = e.Error.Message
		}
	}
	if len(msg) > 2000 {
		msg = msg[:2000] + "..."
	}
	return msg
}

// sseWriter serialises writes from the stream copier and the keepalive goroutine, and
// flushes each one.
type sseWriter struct {
	mu     sync.Mutex
	w      http.ResponseWriter
	cancel context.CancelFunc
	wrote  bool // since the last keepalive check
	failed bool
}

func (s *sseWriter) write(b []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failed {
		return os.ErrClosed
	}
	_, err := s.w.Write(b)
	if err == nil {
		err = http.NewResponseController(s.w).Flush()
	}
	if err != nil {
		s.failed = true
		s.cancel()
		return err
	}
	s.wrote = true
	return nil
}

func (s *sseWriter) flush() { _ = s.write(nil) }

// errorEvent sends an OpenAI-style error event and ends the stream.
func (s *sseWriter) errorEvent(msg string, code int) {
	payload, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": msg, "type": "proxy_error", "code": code},
	})
	_ = s.write([]byte("data: " + string(payload) + "\n\ndata: [DONE]\n\n"))
}

// keepalive sends a comment whenever nothing was written for about an interval, until
// the returned stop is called.
func (s *sseWriter) keepalive(interval time.Duration) (stop func()) {
	quit, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		t := time.NewTicker(interval / 2)
		defer t.Stop()
		for {
			select {
			case <-quit:
				return
			case <-t.C:
				s.mu.Lock()
				idle := !s.wrote
				s.wrote = false
				s.mu.Unlock()
				if idle && s.write([]byte(keepaliveLine)) != nil {
					return
				}
			}
		}
	}()
	return func() { close(quit); <-done }
}
