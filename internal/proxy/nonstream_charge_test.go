package proxy

import (
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// A non-streaming request is charged its whole max_tokens from admission: nothing
// raises its charge while it runs, so two requests whose caps cannot both fit the pool
// do not run together (attacker #6).
func TestNonStreamingChargedItsOutputCap(t *testing.T) {
	var inEngine atomic.Int64
	release := make(chan struct{})
	h := newHarness(t, func(w http.ResponseWriter, r *http.Request) {
		inEngine.Add(1)
		<-release
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":60000,"total_tokens":60003}}`))
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
				`{"model":"m","max_tokens":60000,"messages":[{"role":"user","content":"tenant %d"}]}`, i), nil)
		}(i)
	}
	waitFor(t, "one request in the engine", func() bool { return inEngine.Load() >= 1 })
	time.Sleep(200 * time.Millisecond)
	if n := inEngine.Load(); n != 1 {
		close(release)
		wg.Wait()
		t.Fatalf("%d non-streaming requests with max_tokens 60000 ran together on a 100K pool", n)
	}
	if s := h.p.admission.Snapshot(); s.CacheUsed < 60000 {
		t.Errorf("modelled use %d while a request with max_tokens 60000 runs", s.CacheUsed)
	}
	close(release)
	wg.Wait()
}

// Without max_tokens, a non-streaming request is charged default_output_tokens even
// when its conversation's history says it usually writes less.
func TestNonStreamingWithoutCapChargedDefault(t *testing.T) {
	q, _ := newCacheQueuePool(t, 1<<20, nil)
	ar := AdmissionRequest{Session: "s", PromptTokens: 100, Buffered: true}
	q.mu.Lock()
	q.session("s", q.now()).outputs = []int{10, 10, 10}
	w := &admissionWaiter{session: "s", prompt: 100, buffer: ar.Buffered}
	got := q.chargeLocked(w)
	q.mu.Unlock()
	if w.expOut != q.cfg.Cache.DefaultOutputTokens {
		t.Fatalf("expected output %d (charge %d), want default_output_tokens %d", w.expOut, got, q.cfg.Cache.DefaultOutputTokens)
	}
}
