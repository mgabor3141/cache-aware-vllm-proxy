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

	unlabelled bool // gauges without labels
	dp         bool // a second engine's series
	noRunning  bool // no num_requests_running gauge
	frozen     bool // the usage gauge never moves
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
		used := e.used()
		if e.frozen {
			used = e.base
		}
		labels := []string{`{engine="0"}`}
		if e.unlabelled {
			labels = []string{""}
		}
		if e.dp {
			labels = append(labels, `{engine="1"}`)
		}
		for _, l := range labels {
			fmt.Fprintf(w, "vllm:kv_cache_usage_perc%s %g\n", l, float64(used)/float64(e.blocks-1))
			if !e.noRunning {
				fmt.Fprintf(w, "vllm:num_requests_running%s %d\n", l, len(e.reqs))
			}
			fmt.Fprintf(w, "vllm:num_requests_waiting%s 0\n", l)
		}
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

// Unlabelled gauges are read like labelled ones, so every check still runs.
func TestCalibrateUnlabelledMetrics(t *testing.T) {
	e := &fakeEngine{bs: 16, blocks: 4000, base: 300, ovh: 5, reqs: map[int]*fakeReq{}, unlabelled: true}
	srv := httptest.NewServer(e)
	defer srv.Close()
	var out bytes.Buffer
	if err := calibrate(&Config{Upstream: srv.URL}, &out); err != nil || !strings.Contains(out.String(), "decode_overhead_blocks: 7") {
		t.Fatalf("%v\n%s", err, out.String())
	}
}

// Calibrate refuses rather than report a figure it could not check: a data-parallel
// engine, metrics its checks cannot read, a usage gauge that never moved (review cv-60
// F6, F7, F8).
func TestCalibrateRefuses(t *testing.T) {
	for _, c := range []struct {
		name string
		set  func(*fakeEngine)
		want string
	}{
		{"data-parallel", func(e *fakeEngine) { e.dp = true }, "data-parallel"},
		{"no running gauge", func(e *fakeEngine) { e.noRunning = true }, "vllm:num_requests_running"},
		{"frozen gauge", func(e *fakeEngine) { e.frozen = true }, "never rose"},
	} {
		t.Run(c.name, func(t *testing.T) {
			e := &fakeEngine{bs: 16, blocks: 4000, base: 300, ovh: 5, reqs: map[int]*fakeReq{}}
			c.set(e)
			srv := httptest.NewServer(e)
			defer srv.Close()
			var out bytes.Buffer
			err := calibrate(&Config{Upstream: srv.URL}, &out)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("err %v, want one mentioning %q\n%s", err, c.want, out.String())
			}
			if strings.Contains(out.String(), "overhead_blocks:") {
				t.Fatalf("reported a figure:\n%s", out.String())
			}
		})
	}
}
