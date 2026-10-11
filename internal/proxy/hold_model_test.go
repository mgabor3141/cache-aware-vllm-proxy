package proxy

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"
)

// A finished turn's hold is charged when it finishes, even while a side call of the same
// session still runs, so its room is not sold to other work (l3-93 #1).
func TestHoldChargedAtFinish(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	main := q.start("a", 0, 500000)
	rm := mustAdmit(t, main, "a's main turn")
	side := q.start("a", 0, 3200)
	rs := mustAdmit(t, side, "a's side call")
	main.finish(rm, 500000, 0)
	if h := q.sessionHeldAny("a"); h != 502400 {
		t.Errorf("held %d while the side call runs, want 502400", h)
	}
	mustWait(t, q.start("x", 0, 500000), "x in the room a's finished turn left")
	side.finish(rs, 3200, 0)
	if m := q.metrics(); m.usedTokens > m.budgetTokens {
		t.Errorf("used %d over the pool %d", m.usedTokens, m.budgetTokens)
	}
}

// A hold displaced while its session runs a side call is gone for good: the side call's
// finish leaves its own small hold, not the displaced one (l3-93 #2).
func TestDisplacedHoldNeverReturns(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 5, 300000)
	ta.finish(mustAdmit(t, ta, "a"), 300000, 0)
	clk.add(time.Hour)
	side := q.start("a", 5, 2000)
	rs := mustAdmit(t, side, "a's side call")
	mustAdmit(t, q.start("f", 5, 200000), "filler")
	x := q.start("x", 0, 100000)
	rx := mustAdmit(t, x, "x displacing a's hold")
	if x.rep.DisplacedTokens != 300800 {
		t.Fatalf("displaced %d, want a's 300800", x.rep.DisplacedTokens)
	}
	side.finish(rs, 2000, 10)
	if h := q.sessionHeldAny("a"); h != 3200 {
		t.Errorf("a holds %d after its side call, want the side call's own 3200", h)
	}
	if m := q.metrics(); m.usedTokens > m.budgetTokens {
		t.Errorf("used %d over the pool %d", m.usedTokens, m.budgetTokens)
	}
	rx()
}

// A hold is no larger than the prompt admission priced, whatever the engine reports
// (l3-93 #4).
func TestHoldBoundedByPricedPrompt(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	tr := q.start("a", 0, 200000)
	tr.finish(mustAdmit(t, tr, "a"), 400000, 0)
	if h := q.sessionHeldAny("a"); h != 201600 {
		t.Errorf("held %d, want the priced 201600", h)
	}
}

// A smaller turn that ends without usage keeps the previous hold even when it streamed
// output (l3-92 F2); with usage, a side call that cannot be told apart keeps it too.
func TestTurnWithoutUsageKeepsHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	t1 := q.start("s", 0, 300000)
	t1.finish(mustAdmit(t, t1, "turn 1"), 300000, 0)
	gen := q.NewOutputCounter()
	done := make(chan func())
	go func() {
		rel, _, _ := q.Acquire(context.Background(), AdmissionRequest{Session: "s", PromptTokens: 3000, Generated: gen})
		done <- rel
	}()
	rel := <-done
	gen.Chunks(1)
	rel()
	if h := q.sessionHeldAny("s"); h != 300800 {
		t.Errorf("held %d after a no-usage turn that streamed, want 300800", h)
	}
}

// A release with nothing to show does not re-arm the hold's firmness (l3-94 #4).
func TestUnobservedReleaseKeepsFirmness(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	t1 := q.start("a", 0, 200000)
	t1.finish(mustAdmit(t, t1, "a"), 200000, 0)
	clk.add(10 * time.Minute)
	mustAdmit(t, q.start("a", 0, 1), "ping")()
	q.mu.Lock()
	firm := q.firm(q.sessions["a"], q.now())
	q.mu.Unlock()
	if firm {
		t.Error("an unobserved release made the hold firm again")
	}
}

func (q *AdmissionQueue) sessionHeldAny(key string) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	if s := q.sessions[key]; s != nil {
		return s.held
	}
	return 0
}

// Random sequences of arrivals, finishes (reported, over- and under-reported, without
// usage), streamed output and time, checked after every step against the invariants in
// docs/design.md, "The hold model".
func TestHoldModelProperties(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		t.Run(fmt.Sprint(seed), func(t *testing.T) { holdModelRun(t, seed) })
	}
}

type pmReq struct {
	w      *admissionWaiter
	tokens []int64
	gen    *OutputCounter
	obs    CacheObservation
	ok     bool
	side   bool // a side call: fresh tokens, not the conversation
	step   int  // when it arrived
}

func holdModelRun(t *testing.T, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	q, clk := newCacheQueuePool(t, 320000, nil)
	sessions := []string{"s0", "s1", "s2", "s3", "s4"}
	conv := map[string][]int64{}
	held := map[string][]int64{} // the tokens behind each hold, as the test saw them
	displaced := map[string]bool{}
	heldStep := map[string]int{} // when the test last saw each hold made
	var waiting, running []*pmReq
	next := int64(1)
	fresh := func(n int) []int64 {
		p := make([]int64, n)
		for i := range p {
			p[i] = next
			next++
		}
		return p
	}
	type pre struct {
		held, prio int
		firm       bool
	}
	var before map[string]pre
	// snap records each hold before a step, firm as the grant pass will see it: after the
	// clock moved, and with a finishing request's slot already returned (fewer running
	// requests shorten the firm window).
	snap := func(released int) {
		q.mu.Lock()
		defer q.mu.Unlock()
		q.active -= released
		before = map[string]pre{}
		for k, s := range q.sessions {
			before[k] = pre{s.held, s.priority, q.firm(s, q.now())}
		}
		q.active += released
	}
	for step := 0; step < 300; step++ {
		snap(0)
		finished := ""
		switch op := rng.Intn(10); {
		case op < 4: // arrival: the next turn, a divergent prompt, or a side call
			s := sessions[rng.Intn(len(sessions))]
			var p []int64
			switch r := rng.Intn(4); {
			case r < 2:
				conv[s] = append(append([]int64(nil), conv[s]...), fresh(rng.Intn(20000)+1)...)
				p = conv[s]
			case r == 2 || len(conv[s]) > 150000:
				conv[s] = fresh(rng.Intn(120000) + 1)
				p = conv[s]
			default:
				p = fresh(rng.Intn(5000) + 1)
			}
			pr := &pmReq{tokens: p, gen: q.NewOutputCounter(), side: len(p) <= 5000 && len(p) != len(conv[s]), step: step}
			pr.w, _, _ = q.enqueue(AdmissionRequest{Session: s, Priority: rng.Intn(3), PromptTokens: len(p), Tokens: p,
				MaxTokens: 4000, Generated: pr.gen})
			waiting = append(waiting, pr)
		case op < 7 && len(running) > 0: // a finish
			i := rng.Intn(len(running))
			pr := running[i]
			running = append(running[:i], running[i+1:]...)
			switch rng.Intn(4) {
			case 0:
				pr.obs, pr.ok = CacheObservation{PromptTokens: len(pr.tokens) * 2, CompletionTokens: 10}, true
			case 1:
				pr.obs, pr.ok = CacheObservation{PromptTokens: len(pr.tokens), CompletionTokens: 100}, true
			case 2:
				pr.gen.Chunks(1)
			}
			finished = pr.w.session
			q.mu.Lock()
			s := q.sessions[finished]
			h0, b0 := s.held, s.heldBlocks.Blocks
			q.mu.Unlock()
			snap(1)
			q.release(pr.w, func() (CacheObservation, bool) { return pr.obs, pr.ok })
			q.mu.Lock()
			if s.held > 0 && (s.held != h0 || len(s.heldBlocks.Blocks) != len(b0) || (len(b0) > 0 && &s.heldBlocks.Blocks[0] != &b0[0])) {
				held[finished] = pr.tokens // the finish replaced the hold
				if pr.side && h0 > q.roundUp(len(pr.tokens)) && pr.step < heldStep[finished] {
					t.Fatalf("seed %d step %d: a side call replaced the %d-token hold of a turn that finished after it arrived", seed, step, h0)
				}
				heldStep[finished] = step
				displaced[finished] = false
			}
			q.mu.Unlock()
		case op < 8 && len(running) > 0: // output streams
			pr := running[rng.Intn(len(running))]
			for n := rng.Intn(3000); n > 0; n-- {
				pr.gen.Chunks(1)
			}
			q.mu.Lock()
			q.grantLocked()
			q.mu.Unlock()
		default:
			clk.add(time.Duration(rng.Intn(30)) * time.Second)
			snap(0)
			q.tick()
		}
		q.mu.Lock()
		var granted []*admissionWaiter
		for i := 0; i < len(waiting); i++ {
			if waiting[i].w.granted {
				granted = append(granted, waiting[i].w)
				running = append(running, waiting[i])
				waiting = append(waiting[:i], waiting[i+1:]...)
				i--
			}
		}
		used := q.cacheUsedLocked(q.now())
		trace := fmt.Sprintf("seed %d step %d", seed, step)
		if len(granted) > 0 && used > q.budget() {
			t.Fatalf("%s: modelled use %d over the budget %d at a grant", trace, used, q.budget())
		}
		// Recount from the test's own view: each hold once, each running request less the
		// whole blocks its tokens share with its session's held tokens.
		want := 0
		for _, s := range q.sessions {
			want += s.held
		}
		for _, pr := range running {
			sh := 0
			if s := q.sessions[pr.w.session]; s != nil && s.held > 0 {
				h := held[pr.w.session]
				n := 0
				for n < len(h) && n < len(pr.tokens) && h[n] == pr.tokens[n] {
					n++
				}
				sh = min(min(n, len(pr.tokens)-1)/3200*3200, s.held)
			}
			want += q.liveCharge(pr.w) - sh
		}
		if used != want {
			t.Fatalf("%s: modelled use %d, recount %d: something is counted twice or not at all", trace, used, want)
		}
		for k, b := range before {
			s := q.sessions[k]
			if b.held == 0 || (s != nil && s.held > 0) || k == finished {
				continue
			}
			displaced[k] = true
			ok := false
			for _, w := range granted {
				ok = ok || w.session == k || w.priority < b.prio || !b.firm
			}
			if !ok {
				t.Fatalf("%s: %s firm hold (priority %d) displaced", trace, k, b.prio)
			}
		}
		for k, d := range displaced {
			if s := q.sessions[k]; d && s != nil && s.held > 0 && k != finished {
				t.Fatalf("%s: %s's displaced hold came back (%d)", trace, k, s.held)
			}
		}
		q.mu.Unlock()
	}
	for _, pr := range running {
		q.release(pr.w, func() (CacheObservation, bool) { return CacheObservation{}, false })
	}
}

// A firm hold is not displaced by equal-priority work even with nothing running; once its
// window passes, the waiter gets in (l3-92 F1).
func TestFirmHoldKeptWithEngineIdle(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta, tb := q.start("a", 0, 245000), q.start("b", 0, 245000)
	ra, rb := mustAdmit(t, ta, "a"), mustAdmit(t, tb, "b")
	ta.finish(ra, 245000, 0)
	tb.finish(rb, 245000, 0)
	c := q.start("c", 0, 200000)
	mustWait(t, c, "c while a's and b's holds are firm")
	clk.add(2 * time.Minute)
	q.tick()
	mustAdmit(t, c, "c once the holds are expendable")()
}
