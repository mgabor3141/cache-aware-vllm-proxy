package proxy

import (
	"bytes"
	"context"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

func preempted(v float64) map[string]float64 { return map[string]float64{`{engine="0"}`: v} }

func TestGuardRaiseCooldownCapAndDecay(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	base := q.overheadLocked(true)
	q.EngineSignals(preempted(3)) // first read of the counter: no signal
	if q.margin != 0 {
		t.Fatalf("margin %d after the first counter read", q.margin)
	}
	q.EngineSignals(preempted(4))
	if q.margin != guardStepBlocks || q.overheadLocked(true) != base+guardStepBlocks {
		t.Fatalf("margin %d after a preemption", q.margin)
	}
	q.EngineSignals(preempted(5)) // within the cooldown: one episode, one step
	if q.margin != guardStepBlocks {
		t.Fatalf("margin %d: raised twice inside the cooldown", q.margin)
	}
	for i := 0; i < 40; i++ { // one raise per cooldown
		clk.add(guardCooldown)
		q.EngineSignals(preempted(float64(6 + i)))
	}
	if q.margin != guardMaxMarginBlocks {
		t.Fatalf("margin %d, want the cap %d", q.margin, guardMaxMarginBlocks)
	}
	clk.add(guardDecayAfter - time.Second)
	q.mu.Lock()
	q.guardDecayLocked(q.now())
	q.mu.Unlock()
	if q.margin != guardMaxMarginBlocks {
		t.Fatal("decayed before guardDecayAfter without a signal")
	}
	clk.add(time.Second)
	q.mu.Lock()
	q.guardDecayLocked(q.now())
	q.mu.Unlock()
	if q.margin != guardMaxMarginBlocks-guardStepBlocks {
		t.Fatalf("margin %d after guardDecayAfter without a signal", q.margin)
	}
}

// The counter is judged per series: a series appearing (a data-parallel engine, a
// relabel) or going down (an engine restart) sets a baseline and raises nothing; a rise
// in any series is a preemption. A read with no counter at all is logged, not ignored
// (review cv-60 F9).
func TestGuardPreemptionSeries(t *testing.T) {
	var logs bytes.Buffer
	log.SetOutput(&logs)
	defer log.SetOutput(os.Stderr)
	q, clk := newCacheQueue(t, nil)
	two := func(a, b float64) map[string]float64 { return map[string]float64{`{engine="0"}`: a, `{engine="1"}`: b} }
	q.EngineSignals(preempted(5))
	q.EngineSignals(two(5, 3)) // a second series appears
	q.EngineSignals(two(0, 3)) // engine 0 restarted
	q.EngineSignals(two(0, 3))
	if q.margin != 0 {
		t.Fatalf("margin %d without a preemption", q.margin)
	}
	q.EngineSignals(two(0, 4))
	if q.margin != guardStepBlocks {
		t.Fatalf("margin %d after engine 1 preempted", q.margin)
	}
	clk.add(guardCooldown)
	q.EngineSignals(map[string]float64{})
	q.EngineSignals(map[string]float64{})
	if n := strings.Count(logs.String(), "no vllm:num_preemptions_total"); n != 1 {
		t.Fatalf("missing counter logged %d times, want once:\n%s", n, logs.String())
	}
	q.EngineSignals(two(0, 9)) // the counter is back: a baseline, not a rise
	if q.margin != guardStepBlocks {
		t.Fatalf("margin %d after the counter reappeared", q.margin)
	}
}

// The engine evicting a hold the proxy still protected is not a signal: it drops cached
// prefixes for its own reasons, and with idle holds filling the pool the old near-budget
// gate let it raise the margin with nothing running (review cv-60 F1, F5).
func TestGuardIgnoresHoldEviction(t *testing.T) {
	q, _ := newCacheQueue(t, func(a *AdmissionConfig) {
		a.Cache.PrefillOverheadBlocks, a.Cache.DecodeOverheadBlocks = intPtr(2), intPtr(2)
	})
	m := q.mirror
	m.SetConnected(true)
	mustAdmit(t, q.start("big", 0, 600000), "big")
	prompt := testPrompt(20, 0)
	storePrompt(m, -1, prompt, 1)
	tr := q.startReq("agent", 0, len(prompt), kvmirror.PromptBlockKeys(prompt, 3200))
	tr.finish(mustAdmit(t, tr, "agent"), len(prompt), 100) // a firm hold, near the budget
	q.tick()
	g := 5
	var rm []any
	for i := 0; i < 20; i++ {
		rm = append(rm, uint64(i+1))
	}
	m.Apply(-1, []kvmirror.Event{{Type: "BlockRemoved", BlockHashes: rm, GroupIdx: &g}})
	q.tick()
	if q.evictedHolds != 1 || q.margin != 0 {
		t.Fatalf("evicted %d holds, margin %d: want the hold dropped and no raise", q.evictedHolds, q.margin)
	}
}

// The guard's late signal and the usage figure come from /metrics; a format change must not
// silently disable them.
func TestFetchEngineUsageParses(t *testing.T) {
	for name, c := range map[string]struct {
		body string
		pre  map[string]float64
	}{
		"labelled": {"# HELP x\nvllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.25\n" +
			"vllm:num_preemptions_total{engine=\"0\",model_name=\"m\"} 2.0\n" +
			"vllm:num_preemptions_total{engine=\"1\",model_name=\"m\"} 1.0\n" +
			"vllm:num_preemptions_total_created{engine=\"0\"} 1.7e9\n",
			map[string]float64{`{engine="0",model_name="m"}`: 2, `{engine="1",model_name="m"}`: 1}},
		"unlabelled": {"vllm:kv_cache_usage_perc 0.25\nvllm:num_preemptions_total 3\n", map[string]float64{"": 3}},
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(c.body)) }))
			defer srv.Close()
			u, pre, ok := fetchEngineUsage(context.Background(), srv.URL)
			if !ok || u != 0.25 || len(pre) != len(c.pre) {
				t.Fatalf("usage %v preemptions %v ok %v", u, pre, ok)
			}
			for k, v := range c.pre {
				if pre[k] != v {
					t.Fatalf("preemptions %v, want %v", pre, c.pre)
				}
			}
		})
	}
}

// A raised margin applies to waiters already queued: their charge is priced when they are
// granted, not frozen at arrival, so the queue never over-commits the model by the margin
// (reviews cv-51 MED-1, cv-54 #7, cv-60 F2). And a waiter queued during a raise is not held
// back by it once the margin decays.
func TestGuardMarginReachesQueuedWaiters(t *testing.T) {
	q, clk := newCacheQueuePool(t, 320000, func(a *AdmissionConfig) { // 100 blocks
		a.Cache.PrefillOverheadBlocks, a.Cache.DecodeOverheadBlocks = intPtr(0), intPtr(0)
	})
	const bs = 3200
	blocks := func(n int) int { return n*bs - 4000 } // the fixture reserves 4000 output tokens
	aRel := mustAdmit(t, q.start("a", 0, blocks(40)), "a")
	mustAdmit(t, q.start("b", 0, blocks(58)), "b")
	early := q.start("early", 0, blocks(40))
	mustWait(t, early, "early")

	q.EngineSignals(preempted(3))
	q.EngineSignals(preempted(4)) // the guard raises while early waits
	if got := waiterCharge(q.Residency(), "early"); got != (40+guardStepBlocks)*bs {
		t.Fatalf("queued charge %d, want the raised %d", got, (40+guardStepBlocks)*bs)
	}
	aRel() // frees exactly the pre-raise charge: early no longer fits beside b
	q.tick()
	mustWait(t, early, "early, short by the margin")

	clk.add(guardDecayAfter + time.Second)
	q.mu.Lock()
	q.guardDecayLocked(q.now())
	q.mu.Unlock()
	q.tick()
	rel := mustAdmit(t, early, "early once the margin decayed")
	if r := q.Residency(); r.RunningTokens > r.BudgetTokens {
		t.Fatalf("over-committed: running %d of %d", r.RunningTokens, r.BudgetTokens)
	}
	rel()
}

func waiterCharge(r ResidencySnapshot, session string) int {
	for _, w := range r.Waiters {
		if w.Session == session {
			return w.Charge
		}
	}
	return -1
}
