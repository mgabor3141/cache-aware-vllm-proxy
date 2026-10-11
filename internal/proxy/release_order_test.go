package proxy

import (
	"net/http"
	"testing"
)

// The request's usage is recorded before its permit is released, on both paths:
// admission learns the turn's size at release, and the session's history shows it
// (maintainability #5).
func TestUsageRecordedBeforeRelease(t *testing.T) {
	for _, tc := range []struct {
		name, body       string
		engine           http.HandlerFunc
		prompt, complete int
	}{
		{"stream", streamBody, sseEngine(sseChunk, sseUsage, sseDone), 40, 2},
		{"buffered", chatBody, func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(realVLLMChatResponse))
		}, 78, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.engine, nil)
			if code, body := h.do("POST", "/v1/chat/completions", tc.body, nil); code != 200 {
				t.Fatalf("%d %s", code, body)
			}
			h.event(1)
			q := h.p.admission
			q.mu.Lock()
			defer q.mu.Unlock()
			if len(q.sessions) != 1 {
				t.Fatalf("%d sessions", len(q.sessions))
			}
			for _, s := range q.sessions {
				if s.held != q.roundUp(tc.prompt) || len(s.outputs) != 1 || s.outputs[0] != tc.complete {
					t.Fatalf("session history held=%d outputs=%v, want %d and [%d]: usage reached admission after the release",
						s.held, s.outputs, q.roundUp(tc.prompt), tc.complete)
				}
			}
		})
	}
}
