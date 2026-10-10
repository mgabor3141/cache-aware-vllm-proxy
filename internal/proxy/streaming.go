// streaming.go — keepalive-aware handler for SSE chat completions.
//
// Why this exists separately from the ReverseProxy path:
//
//	httputil.ReverseProxy doesn't write any bytes to the client until
//	the upstream connection is established and starts responding.
//	That's correct for normal traffic but breaks the retry budget
//	we want during a vLLM restart: while the retrying transport is
//	patiently dialing port 8010 every 5-60s for 4+ minutes, the
//	client connection sits silent. App-layer timeouts in browsers
//	and SDKs (30s-2min typical for streaming) fire long before our
//	retries succeed, and the user sees "stream failed".
//
//	The fix for streaming requests is to write SSE headers
//	immediately + start emitting `: keepalive` SSE comment lines on
//	a timer. SSE comments are silently ignored by clients (per
//	spec), so the keepalives are invisible — they only serve to
//	keep bytes flowing on the TCP connection so client-side
//	timeouts don't fire. When upstream finally responds we stop
//	the keepalive timer and pipe the upstream stream through. The
//	transition is seamless to the client.
//
// What happens if all retries exhaust:
//
//	We've already sent HTTP 200 + SSE headers to the client, so
//	we can't return an HTTP error. Instead we emit one final
//	`data: {"error": ...}` event followed by `data: [DONE]`, which
//	chat UIs render as "an error occurred during this response"
//	rather than as a network failure. Visible but not catastrophic.
//
// Non-streaming requests use the existing ReverseProxy path. No
// equivalent JSON-keepalive exists (you can't send a partial JSON
// object). Non-streaming clients are usually backend services with
// long / unlimited timeouts anyway.
package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// keepaliveInterval is how often we emit SSE comment lines while
// waiting for upstream. 10s is short enough that 30s-class client
// timeouts won't fire, long enough that a typical few-second engine
// startup doesn't get noisy log output.
const keepaliveInterval = 10 * time.Second

// keepaliveLine is the SSE comment line we send during the wait.
// Comments start with `:` and are silently ignored by SSE clients
// per the SSE spec. The text after the colon is purely for human
// log inspection (it never reaches a chat UI).
const keepaliveLine = ": cavp keepalive\n\n"

// serveStreamingWithKeepalive handles a streaming chat-completion
// request with end-to-end retry + keepalive coverage.
//
// Preconditions:
//   - r.Body holds the (already-rewritten) JSON body bytes
//   - r.GetBody returns a fresh reader on each call (set in rewriteBody)
//   - the request's `stream: true` field has already been verified
//
// Postcondition: response is fully written (success or error envelope).
func (p *Proxy) serveStreamingWithKeepalive(w http.ResponseWriter, r *http.Request, ev *Event) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		// Should never happen with the standard server, but be
		// defensive: without Flusher we can't stream at all.
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		ev.Status = http.StatusInternalServerError
		return
	}

	// Send SSE headers + flush so the client sees the connection
	// is alive and we're committed to streaming. After this point
	// we cannot return an HTTP error code; failures surface as
	// SSE error events instead.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Disable buffering at any intermediate proxies (nginx-style)
	// so keepalives reach the client in real time.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	ev.Status = http.StatusOK

	// writer wraps w + flusher under a mutex. The keepalive
	// goroutine and the main read loop both write to it; the mutex
	// guarantees writes don't interleave at the byte level. They
	// can still interleave at message boundaries (one keepalive
	// comment landing between two SSE data events), which is fine:
	// SSE comment lines starting with `:` are silently ignored by
	// clients per the spec.
	wr := &lockedWriter{w: w, flusher: flusher}

	// Keepalive runs for the lifetime of the request: from now
	// until this function returns. We don't stop it after upstream
	// returns 200 because vLLM's first-request-after-restart has
	// ~30-60s of warmup work (Triton JIT, CUDA graph capture)
	// AFTER returning 200 and BEFORE sending the first SSE chunk.
	// During that gap, clients (browsers, OpenAI SDK) hit their
	// read timeouts if no bytes flow. Continuous keepalive
	// guarantees bytes always flow.
	//
	// During normal streaming, vLLM sends chunks every ~50ms, so
	// the 10s keepalive ticker rarely fires; if it does, it
	// injects one harmless comment between data events. The cost
	// is a handful of extra bytes per request.
	stopCh := make(chan struct{})
	doneCh := make(chan struct{})
	defer func() {
		select {
		case <-stopCh:
		default:
			close(stopCh)
		}
		<-doneCh
	}()
	go func() {
		defer close(doneCh)
		ticker := time.NewTicker(keepaliveInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stopCh:
				return
			case <-r.Context().Done():
				return
			case <-ticker.C:
				if err := wr.write([]byte(keepaliveLine)); err != nil {
					// Client disconnected. Stop trying.
					return
				}
			}
		}
	}()

	// Admission happens after SSE headers and the keepalive goroutine are
	// active. A queued streaming client therefore continues receiving bytes
	// while it waits for a request-lifetime permit.
	if p.admission != nil && p.shouldAdmit(r) {
		release, _, wait, snap, admitted := p.admission.Acquire(r.Context(), p.admissionRequest(r, ev))
		ev.QueueMS = wait.Milliseconds()
		ev.RequestBytes = r.ContentLength
		ev.ActiveOnArrival = snap.Active
		ev.QueuedOnArrival = snap.Queued
		ev.CacheUsedOnArrival = snap.CacheUsed
		ev.applyAdmissionReport()
		if !admitted {
			if r.Context().Err() == nil {
				writeSSEError(wr, fmt.Errorf("inference admission queue unavailable"), ev)
			}
			return
		}
		defer release()
	}

	// Build the upstream request. We use http.NewRequestWithContext
	// so r.Context() cancellation (client disconnect) propagates.
	body, err := r.GetBody()
	if err != nil {
		writeSSEError(wr, fmt.Errorf("get request body: %w", err), ev)
		return
	}
	upURL := *p.upstream
	upURL.Path = r.URL.Path
	upURL.RawQuery = r.URL.RawQuery
	upReq, err := http.NewRequestWithContext(r.Context(), r.Method, upURL.String(), body)
	if err != nil {
		writeSSEError(wr, fmt.Errorf("build upstream request: %w", err), ev)
		return
	}
	// Copy headers (with Authorization stripped, matching the
	// Director on the ReverseProxy path).
	for k, vs := range r.Header {
		if strings.EqualFold(k, "Authorization") {
			continue
		}
		for _, v := range vs {
			upReq.Header.Add(k, v)
		}
	}
	upReq.Header.Set("Content-Length", fmt.Sprintf("%d", r.ContentLength))
	upReq.GetBody = r.GetBody // so the retrying transport can replay
	upReq.ContentLength = r.ContentLength

	// Use the same retrying Transport the ReverseProxy uses; this
	// is where the actual backoff + reconnect logic lives. Wrap in
	// a Client (no extra timeout; r.Context() owns cancellation).
	client := &http.Client{Transport: p.rp.Transport}

	resp, err := client.Do(upReq)
	if err != nil {
		writeSSEError(wr, fmt.Errorf("upstream after retries: %w", err), ev)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Upstream returned a non-200 (after retries deemed it
		// non-retryable, e.g. 400). Convert the JSON error body
		// to an SSE error event so the chat UI shows something
		// useful.
		raw, _ := io.ReadAll(resp.Body)
		writeSSEError(wr, fmt.Errorf(
			"upstream HTTP %d: %s", resp.StatusCode, truncate(string(raw), 500)), ev)
		return
	}

	// Pipe upstream SSE bytes to client. We don't alter the stream;
	// vLLM's SSE output is correct as-is and the chat UIs expect the
	// raw chunks. We do tee each buffer through a usageScanner (a
	// passive observer that never blocks or mutates the bytes) so the
	// final usage chunk lands in the log. Recorded via defer so every
	// return path (EOF, client disconnect, mid-stream error) logs
	// whatever usage arrived before it stopped.
	scanner := &usageScanner{chunks: ev.generated}
	defer func() {
		if u, ok := scanner.Usage(); ok {
			ev.PromptTokens = u.Prompt
			// Same field as the non-streaming path. Most agent traffic streams, so
			// without this the recompute measurement would be missing on exactly the
			// requests it exists to describe.
			ev.CachedPromptTokens = u.Cached
			// Whether the engine reported prompt_tokens_details at all. Without this the
			// streaming path, which is most agent traffic, cannot distinguish a real
			// cache miss from an engine that sent no details.
			ev.UsageDetails = u.Details
			ev.CompletionTokens = u.Completion
			ev.TotalTokens = u.Total
		}
	}()
	// Read in modest-sized buffers so vLLM's per-chunk flushing
	// propagates without buffer-coalescing delay.
	buf := make([]byte, 4096)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			scanner.Write(buf[:n])
			if writeErr := wr.write(buf[:n]); writeErr != nil {
				// Client disconnected mid-stream. Stop piping;
				// the deferred resp.Body.Close handles upstream.
				return
			}
		}
		if readErr == io.EOF {
			return
		}
		if readErr != nil {
			// Mid-stream upstream error. We've already written
			// SSE chunks to the client, so we can't recover
			// cleanly; emit an error event and a DONE marker.
			writeSSEError(wr, fmt.Errorf("upstream read: %w", readErr), ev)
			return
		}
	}
}

// writeSSEError emits one SSE `data: {"error":...}` event followed
// by `data: [DONE]`, matching the OpenAI streaming error envelope.
// Chat UIs render this as an in-conversation error message rather
// than a transport failure.
func writeSSEError(wr *lockedWriter, err error, ev *Event) {
	payload, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": err.Error(),
			"type":    "upstream_unavailable",
			"code":    "proxy_retry_exhausted",
		},
	})
	_ = wr.write([]byte("data: " + string(payload) + "\n\n"))
	_ = wr.write([]byte("data: [DONE]\n\n"))
	// We're past the status-write moment; ev.Status was already 200.
	// Record the error in the log instead.
	log.Printf("level=warn msg=%q err=%q path=%s",
		"streaming upstream failed after retries", err.Error(), ev.Path)
}

// lockedWriter serializes writes to w + flushes through flusher.
// Both the keepalive goroutine and the main goroutine call write()
// concurrently for a brief window during shutdown; the mutex
// guarantees writes don't interleave at the byte level.
type lockedWriter struct {
	mu      sync.Mutex
	w       http.ResponseWriter
	flusher http.Flusher
}

func (l *lockedWriter) write(b []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := l.w.Write(b); err != nil {
		return err
	}
	l.flusher.Flush()
	return nil
}

// truncate cuts a string to maxLen, appending "..." if cut. Used to
// avoid dumping huge upstream error bodies into logs.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
