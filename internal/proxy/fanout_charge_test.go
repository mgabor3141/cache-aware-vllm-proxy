package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A streamed chunk carrying several choices counts one token per choice.
func TestStreamChunkCountsEveryChoice(t *testing.T) {
	q := &AdmissionQueue{cfg: AdmissionConfig{Cache: CacheConfig{TokensPerChunk: 1}}}
	ctr := q.NewOutputCounter()
	s := &usageScanner{chunks: ctr}
	for i := 0; i < 4; i++ {
		s.Write([]byte(`data: {"choices":[{"index":0,"delta":{"content":"a"}},{"index":1,"delta":{"content":"b"}},{"index":2,"delta":{"content":"c"}}],"usage":null}` + "\n\n"))
	}
	if ctr.Load() != 12 {
		t.Fatalf("4 chunks x 3 choices counted %d, want 12", ctr.Load())
	}
}

func TestSequencesOf(t *testing.T) {
	for body, want := range map[string]int{
		`{}`:                           1,
		`{"n":1}`:                      1,
		`{"n":4}`:                      4,
		`{"prompt":"a"}`:               1,
		`{"prompt":[1,2,3]}`:           1,
		`{"prompt":["a","b","c"]}`:     3,
		`{"prompt":[[1,2],[3]],"n":2}`: 4,
		`{"prompt":["only"],"n":3}`:    3,
	} {
		dec := json.NewDecoder(strings.NewReader(body))
		dec.UseNumber()
		var m map[string]any
		if err := dec.Decode(&m); err != nil {
			t.Fatal(err)
		}
		if got := sequencesOf(m); got != want {
			t.Errorf("%s: %d sequences, want %d", body, got, want)
		}
	}
}

// Each sequence (choice or batched prompt) is charged its own output, buffered or not.
func TestChargeScalesWithSequences(t *testing.T) {
	q, _ := newCacheQueuePool(t, 1<<20, nil)
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, buffered := range []bool{true, false} {
		one := &admissionWaiter{prompt: 100, maxOut: 1000, buffer: buffered, seqs: 1}
		four := &admissionWaiter{prompt: 100, maxOut: 1000, buffer: buffered, seqs: 4}
		q.chargeLocked(one)
		q.chargeLocked(four)
		if four.expOut != 4*one.expOut {
			t.Errorf("buffered=%v: expected output %d for 4 sequences, %d for one", buffered, four.expOut, one.expOut)
		}
	}
}

// Two buffered n=8 requests whose output cannot fit the pool together do not run together.
func TestBufferedChoicesChargedTogether(t *testing.T) {
	var inEngine atomic.Int64
	release := make(chan struct{})
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		inEngine.Add(1)
		<-release
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":8,"total_tokens":11}}`))
	}, func(c *Config) {
		c.Admission.MaxActive = 4
		c.Admission.Cache.PrefillOverheadBlocks = intPtr(0)
		c.Admission.Cache.DecodeOverheadBlocks = intPtr(0)
	})
	h.p.admission.SetPool(782, 128, 0) // a 100K-token pool
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			h.do("POST", "/v1/chat/completions", fmt.Sprintf(
				`{"model":"m","n":8,"max_tokens":12000,"messages":[{"role":"user","content":"tenant %d"}]}`, i), nil)
		}(i)
	}
	waitFor(t, "one request in the engine", func() bool { return inEngine.Load() >= 1 })
	time.Sleep(200 * time.Millisecond)
	n := inEngine.Load()
	close(release)
	wg.Wait()
	if n != 1 {
		t.Fatalf("%d requests with n=8, max_tokens=12000 ran together on a 100K pool", n)
	}
}
