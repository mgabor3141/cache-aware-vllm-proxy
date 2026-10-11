package proxy

import (
	"context"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Waiting for a tokenize slot does not use up the sizing call's timeout: a request that
// queued past tokenizeTimeout behind busy slots is still tokenized, not priced from the
// byte estimate (attacker #5, scale #10).
func TestTokenizeSlotWaitDoesNotForceTheEstimate(t *testing.T) {
	t.Parallel()
	p, _ := tokenizeProxy(t, func(w http.ResponseWriter, _ *http.Request) {
		io.WriteString(w, `{"tokens":[1,2,3]}`)
	})
	for i := 0; i < cap(p.tokenizeSem); i++ {
		p.tokenizeSem <- struct{}{}
	}
	go func() {
		time.Sleep(tokenizeTimeout + 300*time.Millisecond)
		for i := 0; i < cap(p.tokenizeSem); i++ {
			<-p.tokenizeSem
		}
	}()
	ev := parsedEvent(t, chatBody)
	ar := p.admissionRequest(context.Background(), chatRequest(), ev)
	if !ev.Tokenized || ar.PromptTokens != 3 {
		t.Fatalf("after waiting for a slot: tokenized=%v prompt=%d err=%q", ev.Tokenized, ar.PromptTokens, ev.TokenizeError)
	}
}

// The callers' /tokenize shares the slots of the proxy's own sizing calls: the engine
// never sees more concurrent tokenizations than there are slots (round 2, 76#2).
func TestUserTokenizeSharesTheSlots(t *testing.T) {
	var cur, peak atomic.Int64
	h := newHarnessEngine(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenize" {
			w.Write([]byte("{}"))
			return
		}
		n := cur.Add(1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
		cur.Add(-1)
		io.Copy(io.Discard, r.Body)
		w.Write([]byte(`{"count":3,"tokens":[1,2,3]}`))
	}), nil)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if st, _ := h.do("POST", "/v1/tokenize", `{"model":"m","prompt":"hi"}`, nil); st != 200 {
				t.Errorf("status %d", st)
			}
		}()
	}
	wg.Wait()
	if n := peak.Load(); n > int64(cap(h.p.tokenizeSem)) {
		t.Fatalf("engine saw %d concurrent /tokenize, slots are %d", n, cap(h.p.tokenizeSem))
	}
}
