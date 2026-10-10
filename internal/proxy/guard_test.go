package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

func TestGuardRaiseCooldownCapAndDecay(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	base := q.overheadLocked(true)
	q.EngineSignals(3) // first read of the counter: no signal
	if q.margin != 0 {
		t.Fatalf("margin %d after the first counter read", q.margin)
	}
	q.EngineSignals(4)
	if q.margin != guardStepBlocks || q.overheadLocked(true) != base+guardStepBlocks {
		t.Fatalf("margin %d after a preemption", q.margin)
	}
	q.EngineSignals(5) // within the cooldown: one episode, one step
	if q.margin != guardStepBlocks {
		t.Fatalf("margin %d: raised twice inside the cooldown", q.margin)
	}
	for i := 0; i < 40; i++ { // one raise per cooldown
		clk.add(guardCooldown)
		q.EngineSignals(float64(6 + i))
	}
	if q.margin != guardMaxMarginBlocks {
		t.Fatalf("margin %d, want the cap %d", q.margin, guardMaxMarginBlocks)
	}
	clk.add(guardDecayAfter - time.Second)
	q.mu.Lock()
	q.guardDecayLocked(q.now())
	q.mu.Unlock()
	if q.margin != guardMaxMarginBlocks {
		t.Fatal("decayed before a quiet hour")
	}
	clk.add(time.Second)
	q.mu.Lock()
	q.guardDecayLocked(q.now())
	q.mu.Unlock()
	if q.margin != guardMaxMarginBlocks-guardStepBlocks {
		t.Fatalf("margin %d after a quiet hour", q.margin)
	}
}

// The early signal: the engine evicting a hold the proxy still counted as firm counts only
// near the budget; with room to spare it is the engine's own policy.
func TestGuardFirmHoldEviction(t *testing.T) {
	for _, c := range []struct {
		name     string
		big      int // a running request beside the hold, in prompt tokens
		wantRise bool
	}{{"room to spare", 0, false}, {"near the budget", 600000, true}} {
		t.Run(c.name, func(t *testing.T) {
			q, _ := newCacheQueue(t, func(a *AdmissionConfig) {
				a.Cache.PrefillOverheadBlocks, a.Cache.DecodeOverheadBlocks = intPtr(2), intPtr(2)
			})
			m := kvmirror.New()
			q.SetMirror(m)
			m.SetConnected(true)
			if c.big > 0 {
				mustAdmit(t, q.start("big", 0, c.big), "big")
			}
			prompt := make([]int64, 3200*20)
			for i := range prompt {
				prompt[i] = int64(i%5000 + 1)
			}
			g := 5
			var evs []kvmirror.Event
			var parent any
			for i := 0; i < 20; i++ {
				evs = append(evs, kvmirror.Event{Type: "BlockStored", BlockHashes: []any{uint64(i + 1)}, ParentBlockHash: parent,
					TokenIDs: prompt[i*3200 : (i+1)*3200], BlockSize: 3200, GroupIdx: &g, Kind: "full_attention"})
				parent = uint64(i + 1)
			}
			m.Apply(-1, evs)
			tr := q.startReq("agent", 0, len(prompt), kvmirror.PromptBlockKeys(prompt, 3200))
			rel := mustAdmit(t, tr, "agent")
			tr.finish(rel, len(prompt), 100) // its context is now a firm hold

			var rm []any
			for i := 0; i < 20; i++ {
				rm = append(rm, uint64(i+1))
			}
			m.Apply(0, []kvmirror.Event{{Type: "BlockRemoved", BlockHashes: rm, GroupIdx: &g}})
			mustAdmit(t, q.start("other", 0, 1000), "other")
			if q.evictedHolds != 1 {
				t.Fatalf("evicted %d, want 1", q.evictedHolds)
			}
			if got := q.margin > 0; got != c.wantRise {
				t.Fatalf("margin %d", q.margin)
			}
		})
	}
}

// The guard's late signal and the usage figure come from /metrics; a format change must not
// silently disable them.
func TestFetchEngineUsageParses(t *testing.T) {
	for name, body := range map[string]string{
		"labelled": "# HELP x\nvllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 0.25\n" +
			"vllm:num_preemptions_total{engine=\"0\",model_name=\"m\"} 2.0\n" +
			"vllm:num_preemptions_total{engine=\"1\",model_name=\"m\"} 1.0\n",
		"unlabelled": "vllm:kv_cache_usage_perc 0.25\nvllm:num_preemptions_total 3\n",
	} {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(body)) }))
			defer srv.Close()
			u, pre, ok := fetchEngineUsage(context.Background(), srv.URL)
			if !ok || u != 0.25 || pre != 3 {
				t.Fatalf("usage %v preemptions %v ok %v", u, pre, ok)
			}
		})
	}
}
