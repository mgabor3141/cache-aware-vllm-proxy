package proxy

import (
	"context"
	"fmt"
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

// Equal holds idle since the same instant are displaced in a fixed order, not in map
// order (fairness #7).
func TestDisplacementVictimsDeterministic(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, k := range []string{"c", "a", "b"} {
		q.sessions[k] = &cacheSession{key: k, held: 230400, lastDone: clk.now(), heldSeen: -1}
	}
	w := &admissionWaiter{session: "w", prompt: 300000, maxOut: 4000}
	for i := 0; i < 60; i++ {
		plan, ok := q.fitLocked(w, clk.now().Add(time.Hour), 0, q.grantStateLocked(clk.now().Add(time.Hour)))
		if !ok || len(plan) != 2 || plan[0].s != q.sessions["a"] || plan[1].s != q.sessions["b"] {
			var got []string
			for _, d := range plan {
				got = append(got, d.s.key)
			}
			t.Fatalf("run %d: displaced %v (ok %v), want [a b]", i, got, ok)
		}
	}
}

// A waiter that can never run beside the holds it may not displace reserves nothing, so
// work behind it that fits is admitted (attacker #2, fairness #1).
func TestUnsatisfiableWaiterReservesNothing(t *testing.T) {
	for _, tc := range []struct {
		name   string
		prompt int
	}{
		{"larger than the pool", 900000},
		{"larger than the pool less higher-priority holds", 500000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, _ := newCacheQueue(t, nil)
			th := q.start("h", 0, 300000)
			th.finish(mustAdmit(t, th, "h"), 300000, 0) // a priority-0 hold of 300800
			r1 := mustAdmit(t, q.start("r1", 0, 1000), "r1")
			big := q.start("big", 5, tc.prompt)
			mustWait(t, big, "big")
			cold := q.start("cold", 5, 100000) // fits in the 313600 free
			rel := cold.waitAdmitted(time.Second)
			if rel == nil {
				t.Fatal("cold work that fits is held back by a waiter that can never run")
			}
			rel()
			r1()
		})
	}
}

// A request that shares its session's held prefix and fits inside it takes no room, so other waiters'
// reservations do not hold it back.
func TestCoveredRequestIgnoresReservations(t *testing.T) {
	for _, tc := range []struct {
		name    string
		side    []int64
		covered bool
	}{
		{"same prefix", testPrompt(93, 1)[:6401], true},
		{"other prefix", testPrompt(2, 2), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, _ := newCacheQueue(t, nil)
			p := testPrompt(93, 1)
			ta := q.startReq("a", 0, len(p), p)
			ta.finish(mustAdmit(t, ta, "a"), len(p), 0)
			r1 := mustAdmit(t, q.start("r1", 0, 1000), "r1")
			w1 := q.start("w1", 0, 600000) // waits for r1, reserving more than is free
			mustWait(t, w1, "w1")
			side := q.startReq("a", 1, len(tc.side), tc.side)
			rel := side.waitAdmitted(200 * time.Millisecond)
			if tc.covered && rel == nil {
				t.Fatal("a's side call, covered by its own hold, is held back by w1's reservation")
			}
			if !tc.covered && rel != nil {
				t.Fatal("a side call that shares nothing with a's hold skipped w1's reservation")
			}
			if rel != nil {
				rel()
			}
			r1()
		})
	}
}

// With every slot taken, a grant pass returns before it scores waiters against the
// mirror (scale #2).
func TestGrantPassWithSlotsFullDoesNotScore(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	q.mu.Lock()
	defer q.mu.Unlock()
	w := &admissionWaiter{session: "w", prompt: 3200, tokens: testPrompt(1, 1), ready: make(chan struct{})}
	q.waiters = append(q.waiters, w)
	q.active = q.cfg.MaxActive
	q.grantLocked()
	if w.blocks.BlockSize != 0 {
		t.Error("a pass with no free slot built the waiter's mirror keys")
	}
}

// A grant pass does not scan every hold for every blocked waiter: its work per waiter
// does not grow with the session table (scale #1).
func TestGrantPassBlockedWaitersCheap(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	mustAdmit(t, q.start("r1", 0, 300000), "r1")
	q.mu.Lock()
	defer q.mu.Unlock()
	for i := 0; i < 10; i++ {
		k := fmt.Sprintf("h%d", i)
		q.sessions[k] = &cacheSession{key: k, held: 3200, heldSeen: -1, lastDone: clk.now().Add(-time.Hour)}
	}
	for i := 0; i < 100; i++ {
		q.waiters = append(q.waiters, &admissionWaiter{session: fmt.Sprintf("w%d", i), prompt: 400000,
			maxOut: 4000, enqueued: clk.now(), ready: make(chan struct{})})
	}
	allocs := testing.AllocsPerRun(5, q.grantLocked)
	if len(q.waiters) != 100 {
		t.Fatalf("setup: %d waiters left, want all 100 blocked", len(q.waiters))
	}
	if allocs >= 100 {
		t.Errorf("a pass over 100 blocked waiters made %.0f allocations: it builds displacement candidates per waiter", allocs)
	}
}

// A lower-priority session that keeps an overlapping side call running cannot pin its
// hold against a higher-priority waiter (rev-1 F1).
func TestRunningSessionHoldYieldsToHigherPriority(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 5, 300000)
	ta.finish(mustAdmit(t, ta, "a"), 300000, 0)
	clk.add(time.Hour)
	cur := mustAdmit(t, q.start("a", 5, 2000), "side0")
	x := q.start("x", 0, 500000)
	for i := 0; i < 20; i++ {
		if rel := x.waitAdmitted(20 * time.Millisecond); rel != nil {
			rel()
			cur()
			return
		}
		clk.add(time.Minute)
		next := mustAdmit(t, q.start("a", 5, 2000), "side")
		cur()
		cur = next
		q.tick()
	}
	t.Errorf("x starved behind a's pinned hold")
}
