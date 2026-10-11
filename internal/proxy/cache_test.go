package proxy

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *fakeClock) add(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// newCacheQueue is a queue on a known pool of 3200-token blocks: 222 usable blocks (200 of
// a hybrid model, plus room for 28-block runners) unless mutate sets another size.
func newCacheQueue(t *testing.T, mutate func(*AdmissionConfig)) (*AdmissionQueue, *fakeClock) {
	return newCacheQueuePool(t, 710400, mutate)
}

func newCacheQueuePool(t *testing.T, pool int, mutate func(*AdmissionConfig)) (*AdmissionQueue, *fakeClock) {
	t.Helper()
	cfg := AdmissionConfig{
		MaxActive: 8,
		Cache: CacheConfig{
			KVEventsEndpoint: "tcp://vllm:5557", PrefillTokensPerSecond: 4200, DecodeInterference: 0.94, DefaultOutputTokens: 4000,
		},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	q := NewAdmissionQueue(cfg)
	if err := q.cfg.validate(); err != nil {
		t.Fatal(err)
	}
	q.SetPool(pool/3200+1, 3200, 0)
	clk := &fakeClock{t: time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)}
	q.now = clk.now
	return q, clk
}

type turn struct {
	admitted chan func()
	rep      *AdmissionReport
	mu       sync.Mutex
	obs      CacheObservation
	done     chan struct{}
}

func (q *AdmissionQueue) startReq(session string, prio, prompt int, tokens []int64) *turn {
	tr := &turn{admitted: make(chan func(), 1), rep: &AdmissionReport{}, done: make(chan struct{})}
	ar := AdmissionRequest{
		Priority: prio, Session: session, PromptTokens: prompt, MaxTokens: 4000, Tokens: tokens, Report: tr.rep,
		Observe: func() (CacheObservation, bool) {
			tr.mu.Lock()
			defer tr.mu.Unlock()
			return tr.obs, tr.obs.PromptTokens > 0
		},
	}
	q.mu.Lock()
	seq0 := q.sequence
	q.mu.Unlock()
	go func() {
		defer close(tr.done)
		rel, _, ok := q.Acquire(context.Background(), ar)
		if ok {
			tr.admitted <- rel
		}
	}()
	// Return once Acquire registered the request, so the fake clock cannot move first.
	for {
		q.mu.Lock()
		registered := q.sequence > seq0
		q.mu.Unlock()
		if registered {
			return tr
		}
		time.Sleep(100 * time.Microsecond)
	}
}

func (q *AdmissionQueue) start(session string, prio, prompt int) *turn {
	return q.startReq(session, prio, prompt, nil)
}

func (tr *turn) waitAdmitted(d time.Duration) func() {
	select {
	case rel := <-tr.admitted:
		return rel
	case <-time.After(d):
		return nil
	}
}

func (tr *turn) finish(rel func(), prompt, completion int) {
	tr.mu.Lock()
	tr.obs = CacheObservation{PromptTokens: prompt, CompletionTokens: completion}
	tr.mu.Unlock()
	rel()
}

const settle = 50 * time.Millisecond

func mustAdmit(t *testing.T, tr *turn, what string) func() {
	t.Helper()
	rel := tr.waitAdmitted(time.Second)
	if rel == nil {
		t.Fatalf("%s: not admitted", what)
	}
	return rel
}

func mustWait(t *testing.T, tr *turn, what string) {
	t.Helper()
	if rel := tr.waitAdmitted(settle); rel != nil {
		t.Fatalf("%s: admitted, want waiting", what)
	}
}

// Two 200K threads fit in the 511K cache; a third waits although slots are free.
func TestCacheAdmissionBoundsByCacheNotSlots(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	mustAdmit(t, q.start("a", 0, 200000), "a")
	mustAdmit(t, q.start("b", 0, 200000), "b")
	mustWait(t, q.start("c", 0, 200000), "c")
}

// The returning thread is admitted at once: its own hold covers it.
func TestCacheAdmissionReturningThreadUsesOwnHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 200000), "b")
	ta.finish(ra, 200000, 3000)
	mustWait(t, q.start("c", 0, 200000), "c")
	clk.add(5 * time.Second)
	mustAdmit(t, q.start("a", 0, 206000), "returning a")
}

// Review finding 2: a tiny request with the GPU idle and the cache full of holds is
// admitted within seconds, displacing a block or two.
func TestCacheAdmissionTinyRequestNotStuckBehindHolds(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta, tb := q.start("a", 0, 245000), q.start("b", 0, 245000)
	ra, rb := mustAdmit(t, ta, "a"), mustAdmit(t, tb, "b")
	ta.finish(ra, 245000, 5000)
	tb.finish(rb, 245000, 5000)
	tiny := q.start("chat", 0, 2000)
	for i := 0; i < 5 && tiny.waitAdmitted(settle) == nil; i++ {
		clk.add(time.Second)
		q.tick()
	}
	select {
	case <-tiny.done:
	default:
		t.Fatal("tiny request still waiting after 5 s with nothing running")
	}
}

func TestCacheAdmissionSmallRequestUsesLeftover(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	mustAdmit(t, q.start("a", 10, 200000), "a")
	mustAdmit(t, q.start("b", 10, 200000), "b")
	tr := q.start("claw", 0, 20000)
	mustAdmit(t, tr, "small claw")
	if tr.rep.DisplacedTokens != 0 {
		t.Fatal("displaced holds for a request that fit")
	}
}

// A higher-priority request displaces an idle lower-priority hold at once.
func TestCacheAdmissionPriorityDisplacesIdleHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	ta := q.start("a", 10, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 10, 200000), "b")
	ta.finish(ra, 200000, 3000)
	tr := q.start("claw", 0, 150000)
	mustAdmit(t, tr, "high-priority claw")
	if tr.rep.DisplacedTokens == 0 {
		t.Fatalf("report %+v, want a displacement", tr.rep)
	}
}

// A priority request never displaces running work: it waits, then goes first.
func TestCacheAdmissionPriorityWaitsForRunningAndGoesFirst(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	ta := q.start("a", 10, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 10, 200000), "b")
	tc := q.start("c", 10, 200000)
	mustWait(t, tc, "c")
	tclaw := q.start("claw", 0, 150000)
	mustWait(t, tclaw, "claw while a and b run")
	ta.finish(ra, 200000, 3000)
	mustAdmit(t, tclaw, "claw first after a finishes")
}

// Review finding 1: with the production 30m queue timeout, a large waiter facing an
// active agent (one-minute turns, 10 s tool gaps) and a rolling stream of small requests
// gets in by max_agent_wait.
func TestCacheAdmissionLargeWaiterNotStarvedByBackfill(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	agent := q.start("agent", 0, 200000)
	agentRel := mustAdmit(t, agent, "agent")
	agentPrompt, agentRunFor, agentGap := 200000, 6, 0
	big := q.start("big", 0, 330000)
	mustWait(t, big, "big")
	type run struct {
		tr  *turn
		rel func()
	}
	var smalls []run
	for step := 0; step < 120; step++ { // 20 minutes in 10 s steps
		clk.add(10 * time.Second)
		for _, r := range smalls {
			r.tr.finish(r.rel, 30000, 500)
		}
		smalls = smalls[:0]
		switch {
		case agentRel != nil:
			agentRunFor--
			if agentRunFor == 0 {
				agent.finish(agentRel, agentPrompt, 2000)
				agentRel, agentGap = nil, 1
				agentPrompt += 5000
			}
		case agent == nil || agentGap > 0:
			agentGap--
			if agentGap == 0 {
				agent = q.start("agent", 0, agentPrompt)
			}
		default:
			if rel := agent.waitAdmitted(10 * time.Millisecond); rel != nil {
				agentRel, agentRunFor = rel, 6
			}
		}
		q.tick()
		if rel := big.waitAdmitted(10 * time.Millisecond); rel != nil {
			if waited := time.Duration(step+1) * 10 * time.Second; waited > 11*time.Minute {
				t.Fatalf("big admitted after %s", waited)
			}
			return
		}
		tr := q.start("small", 0, 30000)
		if rel := tr.waitAdmitted(20 * time.Millisecond); rel != nil {
			smalls = append(smalls, run{tr, rel})
		}
	}
	t.Fatal("big never admitted")
}

// A request larger than the whole cache still runs, alone.
func TestCacheAdmissionOversizedRunsAlone(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	mustAdmit(t, q.start("huge", 0, 600000), "oversized")
	mustWait(t, q.start("x", 0, 10000), "anything beside it")
}

// Review finding 4: a turn with no usage (failed upstream) leaves no hold.
func TestCacheAdmissionFailedTurnLeavesNoHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	tr := q.start("x", 0, 350000)
	rel := mustAdmit(t, tr, "x")
	rel() // no observation
	if m := q.metrics(); m.held != 0 || m.usedTokens != 0 {
		t.Fatalf("after a failed turn: %+v", m)
	}
}

// Review finding 5: a side call finishing after the main turn does not shrink the hold.
func TestCacheAdmissionOverlappingTurnsKeepLargest(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	main := q.start("s", 0, 242000)
	rm := mustAdmit(t, main, "main")
	side := q.start("s", 0, 3000)
	rs := mustAdmit(t, side, "side")
	main.finish(rm, 242000, 4000)
	side.finish(rs, 3000, 50)
	if h := q.sessionHeld("s"); h != 243200 {
		t.Fatalf("hold %d, want the main turn's 243200", h)
	}
}

// A request whose prefix the mirror reports cached costs less and goes first.
func TestCacheAdmissionCachedRequestJumpsQueue(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	q.mirror.SetSpeculative(true)
	m := q.mirror
	m.SetConnected(true)
	prompt := make([]int64, 3200*30+10)
	for i := range prompt {
		prompt[i] = int64(i % 5000)
	}
	g := 5
	var evs []kvmirror.Event
	var parent any
	for i := 0; i < 30; i++ {
		evs = append(evs, kvmirror.Event{Type: "BlockStored", BlockHashes: []any{uint64(i + 1)}, ParentBlockHash: parent,
			TokenIDs: prompt[i*3200 : (i+1)*3200], BlockSize: 3200, GroupIdx: &g, Kind: "full_attention"})
		parent = uint64(i + 1)
	}
	m.Apply(-1, evs)
	keys := kvmirror.PromptBlockKeys(prompt, 3200)
	// 30 contiguous blocks, minus the speculative-decoding drop.
	if got := m.Cached(kvmirror.PromptKeys{BlockSize: 3200, Blocks: keys}); got != 92800 {
		t.Fatalf("mirror says %d cached, want 92800", got)
	}

	// Fill the slots so both requests queue.
	q.cfg.MaxActive = 1
	ta := q.start("busy", 0, 1000)
	ra := mustAdmit(t, ta, "busy")
	cold := q.start("cold", 0, 100000)
	clk.add(time.Second)
	hot := q.startReq("hot", 0, len(prompt), prompt)
	mustWait(t, cold, "cold")
	mustWait(t, hot, "hot")
	ta.finish(ra, 1000, 10)
	mustAdmit(t, hot, "hot (cached) before the earlier cold request")
	if hot.rep.PredictedCached != 92800 {
		t.Fatalf("predicted cached %d", hot.rep.PredictedCached)
	}
	mustWait(t, cold, "cold behind hot")
}

func TestShippedConfigValidates(t *testing.T) {
	t.Setenv("CAVP_API_KEY", "test-key")
	cfg, err := LoadConfig("../../config.example.yaml")
	if err != nil {
		t.Fatalf("config.yaml: %v", err)
	}
	if cfg.Admission.Cache.KVEventsEndpoint == "" {
		t.Fatal("shipped config has no kv_events_endpoint")
	}
}

// Fifteen agent threads of 200K+ with tool gaps all finish and nothing exceeds the
// budget. Two such threads fit at once and each keeps its cache for its entry quantum
// (about 5 minutes), so a rotation through 15 takes most of an hour: the bound below is
// that rotation, not a target. Clients that time out meanwhile retry in place.
func TestCacheAdmissionFifteenThreadsAllProgress(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	const threads, turns = 15, 4
	type th struct {
		name           string
		prompt         int
		done           int
		cur            *turn
		rel            func()
		runFor, gapFor int
		waitStart      time.Time
	}
	var ts []*th
	for i := 0; i < threads; i++ {
		ts = append(ts, &th{name: string(rune('a' + i)), prompt: 200000 + i*3000})
	}
	for _, x := range ts {
		x.cur = q.start(x.name, 0, x.prompt)
		x.waitStart = clk.now()
	}
	maxWait, maxRun := time.Duration(0), 0
	for step := 0; step < 5000; step++ {
		clk.add(10 * time.Second)
		q.tick()
		time.Sleep(time.Millisecond)
		finished := 0
		for _, x := range ts {
			switch {
			case x.done == turns:
				finished++
			case x.rel == nil && x.cur != nil:
				select {
				case x.rel = <-x.cur.admitted:
					maxWait = max(maxWait, clk.now().Sub(x.waitStart))
					x.runFor = 6
				default:
				}
			case x.rel != nil:
				x.runFor--
				if x.runFor <= 0 {
					x.cur.finish(x.rel, x.prompt, 3000)
					x.rel, x.cur = nil, nil
					x.done++
					x.prompt += 8000
					x.gapFor = 3
				}
			default:
				x.gapFor--
				if x.gapFor <= 0 && x.done < turns {
					x.cur = q.start(x.name, 0, x.prompt)
					x.waitStart = clk.now()
				}
			}
		}
		m := q.metrics()
		maxRun = max(maxRun, q.runningTokens())
		if finished == threads {
			if maxRun > m.budgetTokens {
				t.Fatalf("running tokens %d exceeded budget %d", maxRun, m.budgetTokens)
			}
			if maxWait >= time.Hour {
				t.Fatalf("a turn waited %s, past an hour", maxWait)
			}
			t.Logf("15 threads x %d turns in %d simulated minutes; longest wait %s; displaced %d tokens",
				turns, step/6, maxWait, m.displaced)
			return
		}
	}
	t.Fatal("threads did not all finish")
}

// Review 3 finding 1: two protected 250K holds fill the cache and nothing runs; a 2K
// chat request with no max_tokens (charged 2K + 8,192 output) gets in at once.
func TestCacheAdmissionTinyRequestBesideProtectedHolds(t *testing.T) {
	q, _ := newCacheQueue(t, func(c *AdmissionConfig) { c.Cache.DefaultOutputTokens = 8192 })
	ta, tb := q.start("a", 0, 240000), q.start("b", 0, 240000)
	ra, rb := mustAdmit(t, ta, "a"), mustAdmit(t, tb, "b")
	ta.finish(ra, 250000, 5000) // holds 249,600 each: 9,600 free
	tb.finish(rb, 250000, 5000)
	tr := &turn{admitted: make(chan func(), 1), rep: &AdmissionReport{}, done: make(chan struct{})}
	go func() {
		rel, _, ok := q.Acquire(context.Background(), AdmissionRequest{Session: "chat", PromptTokens: 2000, Report: tr.rep})
		if ok {
			tr.admitted <- rel
		}
	}()
	mustAdmit(t, tr, "tiny chat beside full protected holds")
	if tr.rep.DisplacedTokens != 0 {
		t.Fatalf("displaced %d", tr.rep.DisplacedTokens)
	}
}

// Review 3 finding 2: a higher priority displaces the smallest hold that covers it.
func TestCacheAdmissionPriorityDisplacesBestFit(t *testing.T) {
	// 200 blocks: big and small leave holds of 63 and 18, busy runs 59, so the claw
	// (61) does not fit in the 59-60 left without displacing one.
	q, _ := newCacheQueuePool(t, 640000, nil)
	big := q.start("big", 10, 200000)
	rbig := mustAdmit(t, big, "big")
	small := q.start("small", 10, 57600)
	rsmall := mustAdmit(t, small, "small")
	mustAdmit(t, q.start("busy", 10, 95000), "busy")
	big.finish(rbig, 200000, 1000)
	small.finish(rsmall, 57600, 1000)
	tr := q.start("claw", -5, 100000)
	mustAdmit(t, tr, "priority claw")
	if tr.rep.DisplacedTokens != 57600 {
		t.Fatalf("displaced %d, want the 57,600 hold", tr.rep.DisplacedTokens)
	}
}

func TestParseEnginePool(t *testing.T) {
	m := "# HELP x\nvllm:num_requests_running{engine=\"0\"} 0.0\n" +
		`vllm:cache_config_info{_block_size_resolved="True",block_size="3200",mamba_block_size="16",num_gpu_blocks="201",num_gpu_blocks_override="None",prefix_match_unit="64"} 1.0` + "\n"
	b, s, u, err := parseEnginePool(strings.NewReader(m))
	if err != nil || b != 201 || s != 3200 || u != 64 {
		t.Fatalf("got %d x %d unit %d, %v", b, s, u, err)
	}
	if _, _, u, _ := parseEnginePool(strings.NewReader(strings.Replace(m, `prefix_match_unit="64"`, `prefix_match_unit="None"`, 1))); u != 0 {
		t.Fatalf("unit None read as %d", u)
	}
	if _, _, _, err := parseEnginePool(strings.NewReader("vllm:x 1\n")); err == nil {
		t.Fatal("want an error without cache_config_info")
	}
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 8})
	if q.budget() != 0 {
		t.Fatal("pool should be unknown before SetPool")
	}
	q.SetPool(201, 3200, 0)
	if q.budget() != 640000 {
		t.Fatalf("budget %d, want 200 blocks x 3200", q.budget())
	}
}

// With no queue timeout, a client that times out and retries keeps its place in line.
func TestCacheAdmissionRetryKeepsWait(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	mustAdmit(t, q.start("a", 0, 250000), "a")
	mustAdmit(t, q.start("b", 0, 250000), "b")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		q.Acquire(ctx, AdmissionRequest{Session: "c", PromptTokens: 200000, MaxTokens: 4000, Report: &AdmissionReport{}})
		close(done)
	}()
	time.Sleep(settle)
	clk.add(8 * time.Minute)
	cancel()
	<-done
	clk.add(30 * time.Second)
	tr := q.start("c", 0, 200000)
	time.Sleep(settle)
	if w := q.waiterWait("c"); w < 8*time.Minute {
		t.Fatalf("retry waited %s, want the first attempt's 8m30s", w)
	}
	mustWait(t, tr, "c retry")
}

// A running request's charge follows its streamed output past the expected output, and
// its overhead drops from the prefill to the decode figure once output streams.
func TestCacheAdmissionLiveCharge(t *testing.T) {
	q, _ := newCacheQueue(t, func(c *AdmissionConfig) {
		c.Cache.PrefillOverheadBlocks, c.Cache.DecodeOverheadBlocks = intPtr(28), intPtr(20)
	})
	gen := q.NewOutputCounter()
	q.mu.Lock()
	w := &admissionWaiter{prompt: 100000, expOut: 4000, gen: gen}
	q.running[w] = struct{}{}
	base := q.runningLocked()
	gen.Usage(20000)
	grown := q.runningLocked()
	q.mu.Unlock()
	if base != (33+28)*3200 || grown != (38+20)*3200 {
		t.Fatalf("charges %d %d", base, grown)
	}
}

// An idle hold is firm for its re-prefill time (with the decoding it would stall), then
// expendable: a cold waiter that needs the room takes it.
func TestCacheAdmissionHoldFirmThenExpendable(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 200000), "b")
	ta.finish(ra, 200000, 3000)
	tc := q.start("c", 0, 200000)
	clk.add(60 * time.Second) // firm for 47.6 s x (1 + 0.94 x 1 running) = 92 s
	q.tick()
	mustWait(t, tc, "c while a's hold is firm")
	clk.add(40 * time.Second)
	q.tick()
	mustAdmit(t, tc, "c once a's hold is expendable")
}

// A small hold is still firm for minHold, so a quick tool call returns warm.
func TestCacheAdmissionMinHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 10000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 300000), "b")
	ta.finish(ra, 10000, 500)
	tc := q.start("c", 0, 220000) // needs 2 of a's 3 blocks
	clk.add(2 * time.Second)
	q.tick()
	mustWait(t, tc, "c 2 s after a's turn")
	clk.add(4 * time.Second)
	q.tick()
	mustAdmit(t, tc, "c after minHold")
}

// A thread that is back and waiting keeps its hold firm for the hold window, counted from
// when it started waiting.
func TestCacheAdmissionWaitingOwnerKeepsHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 280000), "b")
	ta.finish(ra, 200000, 3000)
	clk.add(10 * time.Minute)
	ta2 := q.start("a", 0, 260000) // back, but too big to run beside b
	tc := q.start("c", 0, 150000)  // would fit by taking a's hold
	clk.add(10 * time.Second)
	q.tick()
	mustWait(t, ta2, "a")
	mustWait(t, tc, "c, a's owner is waiting")
	if h := q.sessionHeld("a"); h != 63*3200 {
		t.Fatalf("a's hold %d", h)
	}
}

// With the engine busy, an expendable hold is not displaced for cold work: that would
// only add a re-prefill.
func TestCacheAdmissionBusyEngineDoesNotEvict(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 150000)
	ra := mustAdmit(t, ta, "a")
	for _, n := range []string{"b", "c", "d", "e"} {
		mustAdmit(t, q.start(n, 0, 1000), n)
	}
	ta.finish(ra, 150000, 3000)
	tf := q.start("f", 0, 220000) // needs 9 of a's 46 blocks
	clk.add(time.Hour)
	q.tick()
	mustWait(t, tf, "f with four running and only an expendable hold to take")
}

// Order within a priority: a cached thread, then small requests, then cold by wait.
func TestCacheAdmissionTierOrder(t *testing.T) {
	q, clk := newCacheQueue(t, func(c *AdmissionConfig) { c.MaxActive = 1 })
	ta := q.start("a", 0, 100000)
	ra := mustAdmit(t, ta, "a")
	cold := q.start("cold", 0, 100000)
	clk.add(time.Minute)
	small := q.start("small", 0, 5000)
	ta.finish(ra, 100000, 1000)
	mustAdmit(t, small, "small ahead of an older cold waiter")
	mustWait(t, cold, "cold behind small")
}

// Raw /v1/completions has no session key; admission must not dereference a nil session
// (a panic there would leave the queue mutex held and wedge the proxy).
func TestAcquireWithoutSession(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	for i := 0; i < 2; i++ {
		rel, _, ok := q.Acquire(context.Background(), AdmissionRequest{PromptTokens: 2000, MaxTokens: 8, Report: &AdmissionReport{}})
		if !ok {
			t.Fatalf("acquire %d refused", i)
		}
		rel()
	}
}

// storePrompt announces a prompt's full blocks to the mirror, as one attention group.
func storePrompt(m *kvmirror.Mirror, seq int64, prompt []int64, firstHash uint64) {
	g := 5
	var evs []kvmirror.Event
	var parent any
	for i := 0; (i+1)*3200 <= len(prompt); i++ {
		h := firstHash + uint64(i)
		evs = append(evs, kvmirror.Event{Type: "BlockStored", BlockHashes: []any{h}, ParentBlockHash: parent,
			TokenIDs: prompt[i*3200 : (i+1)*3200], BlockSize: 3200, GroupIdx: &g, Kind: "full_attention"})
		parent = h
	}
	m.Apply(seq, evs)
}

func testPrompt(blocks int, salt int64) []int64 {
	p := make([]int64, 3200*blocks+1)
	for i := range p {
		p[i] = int64(i%5000) + 1 + salt*100000
	}
	return p
}

// A hold is judged against the mirror whenever the mirror is healthy, including after an
// engine restart or a resubscribe rebuilt it: there is no exemption for holds seen
// before (reviews cv-51 MED-3, cv-52 #11, cv-60 F4). A hold made while the mirror was
// resyncing is judged once it is healthy again.
func TestCacheAdmissionHoldsRevalidatedAfterMirrorReset(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	m := q.mirror
	m.SetConnected(true)
	prompt := testPrompt(20, 0)
	storePrompt(m, 1, prompt, 1)
	tr := q.startReq("a", 0, len(prompt), prompt)
	tr.finish(mustAdmit(t, tr, "a"), len(prompt), 100)
	q.tick()
	if q.metrics().held == 0 {
		t.Fatal("no hold after the turn")
	}
	m.Apply(2, nil)
	m.Apply(0, nil)      // the engine restarted: the mirror starts over
	m.SetConnected(true) // resubscribed, the replay empty
	q.tick()
	if m := q.metrics(); m.held != 0 || m.evictedHolds != 1 {
		t.Fatalf("hold survived an engine restart: %+v", m)
	}

	// A hold made while the mirror is resyncing is judged once it is healthy.
	m.SetConnected(false)
	storePrompt(m, 1, prompt, 1)
	tr = q.startReq("a", 0, len(prompt), prompt)
	tr.finish(mustAdmit(t, tr, "a"), len(prompt), 100)
	q.tick()
	m.SetConnected(true)
	q.tick()
	g := 5
	var rm []any
	for i := 0; i < 20; i++ {
		rm = append(rm, uint64(i+1))
	}
	m.Apply(2, []kvmirror.Event{{Type: "BlockRemoved", BlockHashes: rm, GroupIdx: &g}})
	q.tick()
	if m := q.metrics(); m.held != 0 || m.evictedHolds != 2 {
		t.Fatalf("evicted hold kept: %+v", m)
	}
}

// Streamed output is counted from the stream's per-chunk usage when it has one, else as
// tokens_per_chunk per choice chunk (default 1): a fixed 4 per chunk over-charged every
// engine without speculative decoding fourfold (reviews cv-51 HIGH-3, cv-54 #6).
func TestOutputCounter(t *testing.T) {
	chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
	for _, c := range []struct {
		name     string
		perChunk int
		stream   string
		want     int64
	}{
		{"default", 0, strings.Repeat(chunk, 3), 3},
		{"mtp", 4, strings.Repeat(chunk, 3), 12},
		{"continuous usage", 4, "data: {\"choices\":[{\"delta\":{\"content\":\"ab\"}}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2,\"total_tokens\":11}}\n\n" +
			"data: {\"choices\":[{\"delta\":{\"content\":\"cde\"}}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":5,\"total_tokens\":14}}\n\n", 5},
		{"final usage", 4, strings.Repeat(chunk, 3) +
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}}\n\ndata: [DONE]\n\n", 3},
	} {
		q, _ := newCacheQueue(t, func(a *AdmissionConfig) { a.Cache.TokensPerChunk = c.perChunk })
		gen := q.NewOutputCounter()
		s := &usageScanner{chunks: gen}
		s.Write([]byte(c.stream))
		if got := gen.Load(); got != c.want {
			t.Errorf("%s: counted %d, want %d", c.name, got, c.want)
		}
	}
}

// A turn that ends without usage (client gone, upstream error) keeps the hold the previous
// turn left: its prefix is what a retry resends (review cv-51 MED-4). One that streamed
// output before it ended is held at the prompt size the proxy measured (review cv-54 #2).
func TestCacheAdmissionAbortedTurnKeepsHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	t1 := q.start("s", 0, 200000)
	t1.finish(mustAdmit(t, t1, "turn 1"), 200000, 3000)
	before := q.metrics().held
	if before == 0 {
		t.Fatal("turn 1 left no hold")
	}
	t2 := q.start("s", 0, 203000)
	mustAdmit(t, t2, "turn 2")() // released without usage
	if got := q.metrics().held; got != before {
		t.Fatalf("held %d after an aborted turn, want the previous %d", got, before)
	}

	gen := q.NewOutputCounter()
	done := make(chan func())
	go func() {
		rel, _, _ := q.Acquire(context.Background(), AdmissionRequest{Session: "t", PromptTokens: 100000, Generated: gen})
		done <- rel
	}()
	rel := <-done
	gen.Chunk()
	rel() // the client left mid-stream: no usage, but output streamed
	if h := q.sessions["t"].held; h != 32*3200 {
		t.Fatalf("held %d, want the measured prompt in whole blocks", h)
	}
}

// A session's hold has the priority of the request that left it, not of the last one
// admitted: a deferred side call does not make the conversation displaceable by everyone
// (review cv-51 MED-6).
func TestCacheAdmissionHoldPriorityFromItsTurn(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	ta := q.start("a", 0, 10000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 300000), "b keeps the engine busy")
	side := q.start("a", 100, 3000)
	rs := mustAdmit(t, side, "deferred side call")
	ta.finish(ra, 10000, 500)
	side.finish(rs, 3000, 50)
	if p := q.sessions["a"].priority; p != 0 {
		t.Fatalf("hold priority %d, want the main turn's 0", p)
	}
	mustWait(t, q.start("c", 0, 220000), "c may not displace a's firm hold")
}

// A full session table forgets only idle sessions that hold nothing, never a session
// with queued requests or a hold (reviews cv-51 MED-12,
// cv-53 #8, cv-58 #15). It prunes below the cap, so it does not rescan per new session.
func TestCacheAdmissionSessionCapSparesHolds(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	q.mu.Lock()
	defer q.mu.Unlock()
	now := clk.now()
	holder := q.session("holder", now.Add(-time.Hour))
	holder.held = 201600
	waiter := q.session("waiter", now.Add(-2*time.Hour))
	waiter.waiting = 1
	for i := 0; len(q.sessions) < sessionCap; i++ {
		q.sessions[fmt.Sprintf("k%05d", i)] = &cacheSession{lastSeen: now.Add(time.Duration(i) * time.Millisecond)}
	}
	q.session("newcomer", now)
	if _, ok := q.sessions["holder"]; !ok {
		t.Fatal("forgot a hold while idle sessions holding nothing remained")
	}
	if _, ok := q.sessions["waiter"]; !ok {
		t.Fatal("forgot a session with a queued request")
	}
	if n := len(q.sessions); n > sessionCap-sessionCap/16+1 {
		t.Fatalf("%d sessions after pruning", n)
	}
	// Parked waiters cannot turn table pressure onto holds: with the table full of
	// sessions that have a queued request, no hold is forgotten (v5-93 #6).
	for k, s := range q.sessions {
		if k != "waiter" && k != "holder" {
			s.waiting = 1
		}
	}
	for i := 0; len(q.sessions) < sessionCap; i++ {
		q.sessions[fmt.Sprintf("m%05d", i)] = &cacheSession{lastSeen: now, waiting: 1}
	}
	q.session("newcomer2", now)
	if _, ok := q.sessions["holder"]; !ok || q.displaced != 0 {
		t.Fatal("a full table of parked waiters forgot a hold")
	}
}

// Test views of the queue's accounting, beside the aggregate one /metrics reads.

func (q *AdmissionQueue) runningTokens() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.runningLocked()
}

func (q *AdmissionQueue) sessionHeld(key string) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	if s := q.sessions[key]; s != nil && s.holding(q.now()) {
		return s.held
	}
	return 0
}

// waiterCharge is what a queued request of the session would be charged now; -1 if none.
func (q *AdmissionQueue) waiterCharge(session string) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, w := range q.waiters {
		if w.session == session {
			return q.chargeLocked(w)
		}
	}
	return -1
}

// waiterWait is how long a queued request of the session counts as having waited.
func (q *AdmissionQueue) waiterWait(session string) time.Duration {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, w := range q.waiters {
		if w.session == session {
			return q.now().Sub(w.enqueued)
		}
	}
	return -1
}

// A hold whose first look sees nothing is gone when the mirror was reset since its turn
// was admitted (here the engine restarted mid-turn), rather than exempt from judgement
// forever. Without a reset in between, a hold the mirror cannot match stays (review 3,
// rev-2 #2).
func TestCacheAdmissionHoldLostDuringItsTurn(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	m := q.mirror
	m.SetConnected(true)
	prompt := testPrompt(20, 0)
	storePrompt(m, 1, prompt, 1)
	tr := q.startReq("a", 0, len(prompt), prompt)
	rel := mustAdmit(t, tr, "a")
	m.Apply(2, nil)
	m.Apply(0, nil)      // the engine restarted while the turn ran
	m.SetConnected(true) // resubscribed, the replay empty
	tr.finish(rel, len(prompt), 100)
	for i := 0; i < 5; i++ {
		clk.add(time.Minute)
		q.tick()
	}
	if got := q.metrics(); got.held != 0 || got.evictedHolds != 1 {
		t.Fatalf("a hold lost to a restart is kept: %+v", got)
	}

	// Control: a prompt the mirror never matches (no reset meanwhile) keeps its hold.
	other := testPrompt(20, 7)
	tr = q.startReq("b", 0, len(other), other)
	tr.finish(mustAdmit(t, tr, "b"), len(other), 100)
	q.tick()
	if q.sessionHeld("b") == 0 {
		t.Fatal("an unmatchable hold was dropped without evidence")
	}
}

// A request tokenized before the engine's block size is known gets its mirror keys once
// it is, so it is scored by what is cached and its hold can be judged (review 3, rev-2 #3).
func TestCacheAdmissionKeysBuiltOnceBlockSizeKnown(t *testing.T) {
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 8, Cache: CacheConfig{KVEventsEndpoint: "tcp://vllm:5557"}})
	q.mirror.SetConnected(true)
	q.mirror.SetSpeculative(true)
	prompt := testPrompt(20, 0)
	rep := &AdmissionReport{}
	got := make(chan func(), 1)
	go func() {
		rel, _, _ := q.Acquire(context.Background(), AdmissionRequest{Session: "a", PromptTokens: len(prompt), Tokens: prompt, Report: rep})
		got <- rel
	}()
	waitForQueued(t, q, 1)
	storePrompt(q.mirror, 1, prompt, 1)
	q.SetPool(223, 3200, 0)
	rel := <-got
	defer rel()
	if rep.PredictedCached != 19*3200 {
		t.Fatalf("predicted cached %d, want %d", rep.PredictedCached, 19*3200)
	}
}

// With a prefix-match unit the engine keeps a prompt to its last unit (with speculative
// decoding, one unit lower), which only the mirror's unit hits show: a two-block hold is
// seen on its first look and dropped when the engine evicts its checkpoint. The session
// keeps a block's worth of unit keys, not one per unit of the prompt.
func TestCacheAdmissionHoldJudgedByUnitHit(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	m := q.mirror
	m.SetSpeculative(true)
	m.SetConnected(true)
	m.SetGeometry(3200, 64)
	prompt := testPrompt(3, 0)[:6900]
	storePrompt(m, 1, prompt, 1)
	tr := q.startReq("a", 0, len(prompt), prompt)
	rel := mustAdmit(t, tr, "a")
	// The engine registers the prompt's end: attention at 6,848, a checkpoint at 6,784.
	g0, g5 := 0, 5
	m.Apply(2, []kvmirror.Event{
		{Type: "BlockStored", BlockHashes: []any{uint64(41)}, ParentBlockHash: uint64(40), TokenIDs: prompt[6720:6784], BlockSize: 64, GroupIdx: &g0, Kind: "mamba"},
		{Type: "BlockStored", BlockHashes: []any{uint64(42)}, ParentBlockHash: uint64(41), TokenIDs: prompt[6784:6848], BlockSize: 64, GroupIdx: &g5, Kind: "full_attention"},
	})
	tr.finish(rel, len(prompt), 100)
	q.mu.Lock()
	s := q.sessions["a"]
	seen, units := s.heldSeen, len(s.heldBlocks.Units)
	q.mu.Unlock()
	if seen != 6784 || units > 3200/64+2 {
		t.Fatalf("hold seen %d with %d unit keys, want 6784 and at most a block's", seen, units)
	}
	m.Apply(3, []kvmirror.Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(41)}, GroupIdx: &g0}})
	q.tick()
	if m := q.metrics(); m.held != 0 || m.evictedHolds != 1 {
		t.Fatalf("evicted hold kept: %+v", m)
	}
}
