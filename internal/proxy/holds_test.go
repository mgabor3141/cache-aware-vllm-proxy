package proxy

import (
	"context"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

// A session's held prefix stays in the budget while one of its own requests runs: the
// engine still has it cached, so the room is not free for others (invariants #1).
func TestHoldChargedWhileOwnRequestRuns(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	ta := q.start("a", 0, 300000)
	ta.finish(mustAdmit(t, ta, "a's turn"), 300000, 0)
	idle := q.metrics().usedTokens
	if idle != 300800 {
		t.Fatalf("setup: used %d, want the 300800-token hold", idle)
	}
	side := q.start("a", 0, 2000)
	rs := mustAdmit(t, side, "a's side call")
	if used := q.metrics().usedTokens; used < idle {
		t.Errorf("while a's side call runs, modelled use fell %d -> %d: the held prefix left the budget", idle, used)
	}
	x := q.start("x", 0, 400000) // fits only if a's hold is forgotten
	mustWait(t, x, "x beside a's running side call and its hold")
	rs()
	q.tick()
}

// A hold the mirror cannot verify (no token IDs: /tokenize failed) has a bounded life: it
// is released once it is no longer firm, even with the engine busy (invariants #2).
func TestUnverifiedHoldReleasedAfterHoldWindow(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	q.mirror.SetConnected(true)
	ta := q.start("a", 0, 200000)
	ta.finish(mustAdmit(t, ta, "a"), 200000, 0)
	q.tick()
	if q.sessionHeld("a") == 0 {
		t.Fatal("setup: a holds nothing")
	}
	var rels []func()
	for _, k := range []string{"r1", "r2", "r3", "r4"} {
		rels = append(rels, mustAdmit(t, q.start(k, 0, 1000), k))
	}
	big := q.start("big", 0, 200000)
	q.tick()
	mustWait(t, big, "big while a's unverified hold is firm")
	clk.add(10 * time.Minute)
	q.tick()
	rel := big.waitAdmitted(time.Second)
	if rel == nil {
		t.Fatalf("big starved: a's unverifiable hold (%d) outlived its hold window", q.sessionHeld("a"))
	}
	rel()
	for _, r := range rels {
		r()
	}
}

// A mirror reset releases a hold whose first look saw nothing, even while it is firm
// (vLLM #9).
func TestUnverifiedHoldReleasedOnMirrorReset(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	q.mirror.SetConnected(true)
	ta := q.start("a", 0, 200000)
	ta.finish(mustAdmit(t, ta, "a"), 200000, 0)
	q.tick()
	if q.sessionHeld("a") == 0 {
		t.Fatal("setup: a holds nothing")
	}
	q.mirror.Apply(-1, []kvmirror.Event{{Type: "AllBlocksCleared"}})
	q.tick()
	if h := q.sessionHeld("a"); h != 0 {
		t.Errorf("a still holds %d after a mirror reset it was never seen through", h)
	}
}

// A waiter that cannot run does not keep its session's idle hold firm forever: firmness
// from a waiting thread lasts the hold window, counted from when it started waiting
// (attacker #1).
func TestParkedWaiterDoesNotPinHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("att", 0, 200000)
	ta.finish(mustAdmit(t, ta, "att's turn"), 200000, 0)
	r1 := mustAdmit(t, q.start("r1", 0, 1000), "r1")
	v := q.start("v", 0, 500000) // needs att's hold displaced
	mustWait(t, v, "v while att's hold is firm")
	parked := q.start("att", 0, 900000) // can never run beside r1
	mustWait(t, parked, "att's oversized request")
	clk.add(time.Hour)
	q.tick()
	rel := v.waitAdmitted(time.Second)
	if rel == nil {
		t.Fatal("v starved an hour: att's parked waiter keeps its hold firm")
	}
	rel()
	r1()
}

// Retry memory carries one attempt's wait, not a chain: a session that gives up and
// returns again and again does not keep moving ahead of honest waiters (attacker #8).
func TestRetryMemoryDoesNotCompound(t *testing.T) {
	q, clk := newCacheQueue(t, func(c *AdmissionConfig) { c.MaxActive = 1 })
	mustAdmit(t, q.start("z", 0, 3200), "blocker")
	for i := 0; i < 6; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			defer close(done)
			if rel, _, ok := q.Acquire(ctx, AdmissionRequest{Session: "pinger", PromptTokens: 20000}); ok {
				rel()
			}
		}()
		waitForQueued(t, q, 1)
		clk.add(10 * time.Second)
		cancel()
		<-done
		clk.add(time.Minute)
	}
	q.start("pinger", 0, 20000)
	if w := q.waiterWait("pinger"); w > 2*time.Minute {
		t.Errorf("pinger credited %v of waiting after six 10 s attempts", w)
	}
}
