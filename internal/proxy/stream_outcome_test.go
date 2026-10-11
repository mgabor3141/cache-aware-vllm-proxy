package proxy

import (
	"bufio"
	"context"
	"net/http"
	"strings"
	"testing"
)

// A stream the engine cut, or one whose client left, is not counted as a success: the
// log and cavp_requests_total carry 502 and 499 (operator #1, chaos #10).
func TestStreamFailuresNotCountedAs200(t *testing.T) {
	t.Run("engine cut", func(t *testing.T) {
		h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
			sseEngine(sseChunk)(w, r)
			conn, _, _ := w.(http.Hijacker).Hijack()
			conn.Close()
		}, nil)
		h.do("POST", "/v1/chat/completions", streamBody, nil)
		if ev := h.event(1); num(ev["status"]) != 502 {
			t.Fatalf("log status %v", ev)
		}
		if _, m := h.do("GET", "/metrics", "", nil); !strings.Contains(m, `cavp_requests_total{route="chat",status="502"} 1`) {
			t.Fatalf("metrics:\n%s", m)
		}
	})
	t.Run("client gone", func(t *testing.T) {
		gone := make(chan struct{})
		h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
			sseEngine(sseChunk)(w, r)
			<-gone
			<-r.Context().Done()
		}, nil)
		ctx, cancel := context.WithCancel(context.Background())
		resp, err := h.request(ctx, "POST", "/v1/chat/completions", streamBody, nil)
		if err != nil {
			t.Fatal(err)
		}
		br := bufio.NewReader(resp.Body)
		for {
			line, err := br.ReadString('\n')
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(line, "data:") {
				break
			}
		}
		cancel()
		resp.Body.Close()
		close(gone)
		if ev := h.event(1); num(ev["status"]) != statusClientGone {
			t.Fatalf("log status %v", ev)
		}
	})
}
