package proxy

import (
	"context"
	"testing"
	"time"
)

// slotQueue is a queue on a known pool large enough that only max_active binds.
func slotQueue(maxActive int) *AdmissionQueue {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: maxActive})
	q.SetPool(1001, 3200, 0)
	return q
}

// Within what fits, the lowest priority value runs first, then the earliest arrival.
func TestAdmissionPriorityOrder(t *testing.T) {
	q := slotQueue(1)
	release, _, ok := q.Acquire(context.Background(), AdmissionRequest{})
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
			r, _, ok := q.Acquire(context.Background(), AdmissionRequest{Priority: prio, PromptTokens: 100000})
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
	q := slotQueue(1)
	release, _, ok := q.Acquire(context.Background(), AdmissionRequest{})
	if !ok {
		t.Fatal("initial acquire failed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan bool)
	go func() {
		_, _, ok := q.Acquire(ctx, AdmissionRequest{Session: "s"})
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
	if w := q.sessions["s"].waiting; w != 0 {
		t.Fatalf("session waiting=%d after its waiter gave up", w)
	}
	release()
}

func TestAdmissionAllowsConfiguredConcurrency(t *testing.T) {
	q := slotQueue(4)
	var releases []func()
	for i := 0; i < 4; i++ {
		r, adm, ok := q.Acquire(context.Background(), AdmissionRequest{})
		if !ok || adm.OnArrival.Active != i {
			t.Fatalf("acquire %d: ok=%v snapshot=%+v", i, ok, adm.OnArrival)
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

// Until the engine's pool is known nothing is admitted; requests wait and run once it is
// read.
func TestAdmissionWaitsForThePool(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 4})
	got := make(chan func(), 1)
	go func() {
		r, _, ok := q.Acquire(context.Background(), AdmissionRequest{Session: "s", PromptTokens: 1000})
		if ok {
			got <- r
		}
	}()
	waitForQueued(t, q, 1)
	select {
	case <-got:
		t.Fatal("admitted before the pool was known")
	case <-time.After(50 * time.Millisecond):
	}
	q.SetPool(201, 3200, 0)
	select {
	case r := <-got:
		r()
	case <-time.After(time.Second):
		t.Fatal("not admitted once the pool was known")
	}
}

// A session's waiting count matches its queued requests on every path: admitted at once,
// admitted after waiting, given up (reviews cv-51 HIGH-2, cv-53 #1, cv-58 #8). A leaked
// count would keep the session's hold firm forever.
func TestAdmissionWaitingCountDrains(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 1})
	waiting := func() int {
		q.mu.Lock()
		defer q.mu.Unlock()
		return q.sessions["s"].waiting
	}
	first := make(chan func(), 1)
	go func() {
		r, _, _ := q.Acquire(context.Background(), AdmissionRequest{Session: "s", PromptTokens: 1000})
		first <- r
	}()
	waitForQueued(t, q, 1) // the pool is unknown: it waits
	if waiting() != 1 {
		t.Fatalf("waiting=%d while queued", waiting())
	}
	q.SetPool(201, 3200, 0)
	rel := <-first
	if waiting() != 0 {
		t.Fatalf("waiting=%d after admission from the queue", waiting())
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		q.Acquire(ctx, AdmissionRequest{Session: "s", PromptTokens: 1000})
		close(done)
	}()
	waitForQueued(t, q, 1) // max_active binds
	cancel()
	<-done
	if waiting() != 0 {
		t.Fatalf("waiting=%d after a waiter gave up", waiting())
	}
	rel()
	r, _, ok := q.Acquire(context.Background(), AdmissionRequest{Session: "s", PromptTokens: 1000})
	if !ok || waiting() != 0 {
		t.Fatalf("waiting=%d after immediate admission", waiting())
	}
	r()
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
