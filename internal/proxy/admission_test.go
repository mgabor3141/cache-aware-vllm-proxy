package proxy

import (
	"context"
	"testing"
	"time"
)

// Slot admission (before the engine pool is known, or with cache admission off) runs
// the lowest priority value first, then the earliest arrival.
func TestSlotAdmissionOrder(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 1})
	release, _, _, _, ok := q.Acquire(context.Background(), AdmissionRequest{})
	if !ok {
		t.Fatal("initial acquire failed")
	}
	type result struct {
		name    string
		release func()
	}
	results := make(chan result, 3)
	enqueue := func(name string, prio, queued int) {
		go func() {
			r, _, _, _, ok := q.Acquire(context.Background(), AdmissionRequest{Priority: prio})
			if ok {
				results <- result{name, r}
			}
		}()
		waitForQueued(t, q, queued)
	}
	enqueue("batch", 10, 1)
	enqueue("early", 0, 2)
	enqueue("late", 0, 3)
	release()
	for _, want := range []string{"early", "late", "batch"} {
		r := <-results
		if r.name != want {
			t.Fatalf("admitted %s, want %s", r.name, want)
		}
		r.release()
	}
}

func TestAdmissionQueueCancellationRemovesWaiter(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 1})
	release, _, _, _, ok := q.Acquire(context.Background(), AdmissionRequest{})
	if !ok {
		t.Fatal("initial acquire failed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool)
	go func() {
		_, _, _, _, ok := q.Acquire(ctx, AdmissionRequest{})
		done <- ok
	}()
	waitForQueued(t, q, 1)
	cancel()
	if <-done {
		t.Fatal("cancelled waiter was admitted")
	}
	if got := q.Snapshot().Queued; got != 0 {
		t.Fatalf("queued=%d, want 0", got)
	}
	release()
}

func TestAdmissionAllowsConfiguredConcurrency(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 4})
	var releases []func()
	for i := 0; i < 4; i++ {
		r, _, _, snap, ok := q.Acquire(context.Background(), AdmissionRequest{})
		if !ok || snap.Active != i {
			t.Fatalf("acquire %d: ok=%v snapshot=%+v", i, ok, snap)
		}
		releases = append(releases, r)
	}
	if got := q.Snapshot().Active; got != 4 {
		t.Fatalf("active=%d, want 4", got)
	}
	for _, release := range releases {
		release()
	}
}

func waitForQueued(t *testing.T, q *AdmissionQueue, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if q.Snapshot().Queued == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("queue did not reach %d", want)
}
