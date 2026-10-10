package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeEngine models a vLLM whose requests each hold ceil((prompt+generated)/bs) attention
// blocks plus a fixed overhead, over a baseline of pinned blocks.
type fakeEngine struct {
	mu                    sync.Mutex
	bs, blocks, base, ovh int
	reqs                  map[int]*fakeReq
	next                  int
}

type fakeReq struct{ prompt, gen int }

func (e *fakeEngine) used() int {
	n := e.base
	for _, r := range e.reqs {
		n += (r.prompt+r.gen+e.bs-1)/e.bs + e.ovh
	}
	return n
}

func (e *fakeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/v1/models":
		w.Write([]byte(`{"data":[{"id":"m"}]}`))
	case "/metrics":
		e.mu.Lock()
		fmt.Fprintf(w, "vllm:cache_config_info{block_size=\"%d\",num_gpu_blocks=\"%d\"} 1.0\n", e.bs, e.blocks)
		fmt.Fprintf(w, "vllm:kv_cache_usage_perc{engine=\"0\"} %g\n", float64(e.used())/float64(e.blocks-1))
		fmt.Fprintf(w, "vllm:num_requests_running{engine=\"0\"} %d\nvllm:num_requests_waiting{engine=\"0\"} 0\n", len(e.reqs))
		e.mu.Unlock()
	case "/v1/completions":
		var body struct {
			Prompt    []int `json:"prompt"`
			MaxTokens int   `json:"max_tokens"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		e.mu.Lock()
		id := e.next
		e.next++
		req := &fakeReq{prompt: len(body.Prompt)}
		e.reqs[id] = req
		e.mu.Unlock()
		fl := w.(http.Flusher)
		time.Sleep(150 * time.Millisecond) // prefill
		for g := 1; g <= body.MaxTokens; g += 16 {
			e.mu.Lock()
			req.gen = g
			e.mu.Unlock()
			fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"x\"}],\"usage\":{\"completion_tokens\":%d}}\n\n", g)
			fl.Flush()
			time.Sleep(15 * time.Millisecond)
		}
		e.mu.Lock()
		delete(e.reqs, id)
		e.base += 1 // a grace pin of the finished turn
		e.mu.Unlock()
		w.Write([]byte("data: [DONE]\n\n"))
	default:
		http.NotFound(w, r)
	}
}

func TestCalibrateMeasuresOverhead(t *testing.T) {
	e := &fakeEngine{bs: 16, blocks: 4000, base: 300, ovh: 5, reqs: map[int]*fakeReq{}}
	srv := httptest.NewServer(e)
	defer srv.Close()
	var out bytes.Buffer
	if err := calibrate(&Config{Upstream: srv.URL}, &out); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	for _, want := range []string{"prefill_overhead_blocks: 7", "decode_overhead_blocks: 7"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("want %q (overhead 5, +2) in:\n%s", want, out.String())
		}
	}
}
