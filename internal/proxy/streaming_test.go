package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const streamBody = `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`

const (
	sseChunk = `data: {"id":"c","choices":[{"index":0,"delta":{"content":"x"}}],"usage":null}` + "\n\n"
	sseUsage = `data: {"id":"c","choices":[],"usage":{"prompt_tokens":40,"completion_tokens":2,"total_tokens":42,"prompt_tokens_details":{"cached_tokens":32}},"kv_transfer_params":null}` + "\n\n"
	sseDone  = "data: [DONE]\n\n"
)

func sseEngine(parts ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, p := range parts {
			io.WriteString(w, p)
			w.(http.Flusher).Flush()
		}
	}
}

func TestStreamRelaysAndRecordsUsage(t *testing.T) {
	var sent atomic.Value
	stream := sseEngine(sseChunk, sseChunk, sseUsage, sseDone)
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		sent.Store(b)
		stream(w, r)
	}, nil)
	code, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	if code != 200 || body != sseChunk+sseChunk+sseUsage+sseDone {
		t.Fatalf("%d %q", code, body)
	}
	so, _ := sent.Load().(map[string]any)["stream_options"].(map[string]any)
	if so["include_usage"] != true {
		t.Errorf("include_usage not set: %v", sent.Load())
	}
	ev := h.event(1)
	if num(ev["status"]) != 200 || ev["error"] != nil || ev["stream"] != true ||
		num(ev["prompt_tokens"]) != 40 || num(ev["cached_prompt_tokens"]) != 32 {
		t.Fatalf("log: %v", ev)
	}
}

// A caller that sets include_usage keeps its value.
func TestStreamIncludeUsageRespected(t *testing.T) {
	var sent atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		sent.Store(string(b))
		sseEngine(sseDone)(w, r)
	}, nil)
	h.do("POST", "/v1/chat/completions", `{"model":"m","stream":true,"stream_options":{"include_usage":false},"messages":[]}`, nil)
	if !strings.Contains(sent.Load().(string), `"include_usage":false`) {
		t.Fatal(sent.Load())
	}
}

// Only a JSON true streams; "stream":"true" is a single response, documented.
func TestStreamNeedsJSONTrue(t *testing.T) {
	var sent atomic.Value
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		sent.Store(string(b))
		w.Write([]byte("{}"))
	}, nil)
	h.do("POST", "/v1/chat/completions", `{"model":"m","stream":"true","messages":[]}`, nil)
	if strings.Contains(sent.Load().(string), "include_usage") || h.event(1)["stream"] != nil {
		t.Fatal("a string stream field took the streaming path")
	}
}

// An engine refusal reaches the client as an error event with the engine's message, and
// the log carries the engine's status (cv-54 #3, cv-59 #2).
func TestStreamEngineErrorLogged(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		io.WriteString(w, `{"object":"error","message":"prompt too long","type":"BadRequestError","code":400}`+strings.Repeat(" ", 1<<20))
	}, nil)
	code, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	if code != 200 || !strings.Contains(body, "prompt too long") || !strings.HasSuffix(body, sseDone) {
		t.Fatalf("%d %q", code, body)
	}
	if ev := h.event(1); num(ev["status"]) != 400 || !strings.Contains(fmt.Sprint(ev["error"]), "prompt too long") {
		t.Fatalf("log: %v", ev)
	}
}

// A transport failure names no internal address to the client (cv-56 #5).
func TestStreamEngineUnreachable(t *testing.T) {
	h := newHarness(t, nil, nil)
	h.eng.Close()
	_, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	if !strings.Contains(body, `"error"`) || strings.Contains(body, "127.0.0.1") {
		t.Fatalf("%q", body)
	}
	if ev := h.event(1); num(ev["status"]) != 502 {
		t.Fatalf("log: %v", ev)
	}
}

// A stream that ends without [DONE] is an error, for the client and in the log (cv-54 #1).
func TestStreamWithoutDone(t *testing.T) {
	h := newHarness(t, sseEngine(sseChunk), nil)
	_, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	if !strings.Contains(body, `"error"`) {
		t.Fatalf("client not told: %q", body)
	}
	if ev := h.event(1); !strings.Contains(fmt.Sprint(ev["error"]), "without [DONE]") {
		t.Fatalf("log: %v", ev)
	}
}

// A stream cut by the engine is an error too.
func TestStreamCutByEngine(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		sseEngine(sseChunk)(w, r)
		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}, nil)
	_, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	if !strings.Contains(body, `"error"`) {
		t.Fatalf("client not told: %q", body)
	}
	if ev := h.event(1); !strings.Contains(fmt.Sprint(ev["error"]), "engine stream cut") {
		t.Fatalf("log: %v", ev)
	}
}

// A streaming client gets keepalives while it waits for the engine, then the stream.
func TestStreamKeepaliveWhileWaiting(t *testing.T) {
	h := newHarness(t, sseEngine(sseChunk, sseDone), nil)
	h.p.keepalive = 40 * time.Millisecond
	h.p.ready.set(false, "test")
	resp, err := h.request(context.Background(), "POST", "/v1/chat/completions", streamBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, len(keepaliveLine))
	if _, err := io.ReadFull(resp.Body, buf); err != nil || string(buf) != keepaliveLine {
		t.Fatalf("%q %v", buf, err)
	}
	h.p.ready.set(true, "")
	rest, _ := io.ReadAll(resp.Body)
	if !strings.HasSuffix(string(rest), sseChunk+sseDone) {
		t.Fatalf("%q", rest)
	}
}

// Keepalives fill silences only: none while chunks flow, one during a long prefill
// (cv-54 #10).
func TestKeepaliveOnlyWhenSilent(t *testing.T) {
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		time.Sleep(300 * time.Millisecond) // prefill
		for i := 0; i < 30; i++ {
			io.WriteString(w, sseChunk)
			w.(http.Flusher).Flush()
			time.Sleep(10 * time.Millisecond)
		}
		io.WriteString(w, sseDone)
	}, nil)
	h.p.keepalive = 100 * time.Millisecond
	_, body := h.do("POST", "/v1/chat/completions", streamBody, nil)
	first := strings.Index(body, "data:")
	if first < 0 || !strings.Contains(body[:first], keepaliveLine) {
		t.Fatalf("no keepalive during the prefill: %q", body)
	}
	if strings.Contains(body[first:], keepaliveLine) {
		t.Fatalf("keepalive between chunks: %q", body)
	}
}
