package proxy

import (
	"context"
	"testing"
	"time"
)

// A returning session's waiter inherits an earlier attempt's place in the queue, but
// the oldest-wait gauge and Admitted.Wait measure this attempt only (rev-3 #2).
func TestOldestWaitIsThisAttempt(t *testing.T) {
	q, clk := newCacheQueuePool(t, 100000, func(c *AdmissionConfig) { c.MaxActive = 1 })
	hold, _, _ := q.Acquire(context.Background(), AdmissionRequest{PromptTokens: 100})
	ctx, cancel := context.WithCancel(context.Background())
	gone := make(chan struct{})
	go func() { q.Acquire(ctx, AdmissionRequest{Session: "s", PromptTokens: 100}); close(gone) }()
	waitFor(t, "first attempt queued", func() bool { return q.Snapshot().Queued == 1 })
	clk.add(time.Hour)
	cancel()
	<-gone
	got := make(chan Admitted, 1)
	go func() {
		rel, adm, _ := q.Acquire(context.Background(), AdmissionRequest{Session: "s", PromptTokens: 100})
		got <- adm
		rel()
	}()
	waitFor(t, "retry queued", func() bool { return q.Snapshot().Queued == 1 })
	clk.add(time.Minute)
	if w := q.metrics().oldestWait; w != time.Minute {
		t.Fatalf("oldest wait %v, want this attempt's 1m", w)
	}
	hold()
	if adm := <-got; adm.Wait != time.Minute {
		t.Fatalf("Admitted.Wait %v, want 1m", adm.Wait)
	}
}
