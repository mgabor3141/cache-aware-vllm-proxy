package proxy

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
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

// Production-shaped config: 30m queue timeout, 10m max wait.
func newCacheQueue(t *testing.T, mutate func(*AdmissionConfig)) (*AdmissionQueue, *fakeClock) {
	t.Helper()
	cfg := AdmissionConfig{
		MaxActive: 8,
		Cache: CacheConfig{
			Enabled: true, Tokens: 710400, BlockTokens: 3200, // 200 usable blocks of a hybrid model, plus room for 28-block runners
			PrefillTokensPerSecond: 4200, DecodeInterference: 0.94, DefaultOutputTokens: 4000,
		},
	}
	if mutate != nil {
		mutate(&cfg)
	}
	q := NewAdmissionQueue(cfg)
	if err := q.cfg.validate(); err != nil {
		t.Fatal(err)
	}
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

func (q *AdmissionQueue) startReq(session string, prio, prompt int, blocks []kvmirror.BlockKey) *turn {
	tr := &turn{admitted: make(chan func(), 1), rep: &AdmissionReport{}, done: make(chan struct{})}
	ar := AdmissionRequest{
		Priority: prio, Session: session, PromptTokens: prompt, MaxTokens: 4000, Blocks: kvmirror.PromptKeys{Blocks: blocks}, Report: tr.rep,
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
		rel, _, _, _, ok := q.Acquire(context.Background(), ar)
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

func (q *AdmissionQueue) tick() {
	q.mu.Lock()
	q.grantLocked()
	q.mu.Unlock()
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
	if r := q.Residency(); r.HeldTokens != 0 || r.UsedTokens != 0 {
		t.Fatalf("after a failed turn: %+v", r)
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
	r := q.Residency()
	if len(r.Sessions) != 1 || r.Sessions[0].Footprint != 246000 {
		t.Fatalf("residency %+v, want footprint 246000", r.Sessions)
	}
}

// A request whose prefix the mirror reports cached costs less and goes first.
func TestCacheAdmissionCachedRequestJumpsQueue(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	m := kvmirror.New()
	q.SetMirror(m)
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
	if got := m.CachedBlockKeys(keys); got != 92800 {
		t.Fatalf("mirror says %d cached, want 92800", got)
	}

	// Fill the slots so both requests queue.
	q.cfg.MaxActive = 1
	ta := q.start("busy", 0, 1000)
	ra := mustAdmit(t, ta, "busy")
	cold := q.start("cold", 0, 100000)
	clk.add(time.Second)
	hot := q.startReq("hot", 0, len(prompt), keys)
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
	if !cfg.Admission.Cache.enabled() {
		t.Fatal("shipped config has cache admission off")
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
		r := q.Residency()
		maxRun = max(maxRun, r.RunningTokens)
		if testing.Verbose() && step%30 == 0 && step < 600 {
			var hs []string
			for _, s := range r.Sessions {
				hs = append(hs, fmt.Sprintf("%s:r%d/h%dK", s.Session, s.Running, s.Held/1000))
			}
			var ws []string
			for _, w := range r.Waiters {
				ws = append(ws, fmt.Sprintf("%s:%ds", w.Session, w.WaitS))
			}
			t.Logf("t=%3dm run=%dK held=%dK %v waiting %v", step/6, r.RunningTokens/1000, r.HeldTokens/1000, hs, ws)
		}
		if finished == threads {
			if maxRun > r.BudgetTokens {
				t.Fatalf("running tokens %d exceeded budget %d", maxRun, r.BudgetTokens)
			}
			if maxWait >= time.Hour {
				t.Fatalf("a turn waited %s, past an hour", maxWait)
			}
			t.Logf("15 threads x %d turns in %d simulated minutes; longest wait %s; displaced %d tokens",
				turns, step/6, maxWait, r.Displaced)
			return
		}
	}
	t.Fatal("threads did not all finish")
}

func TestCallerPriorityHeaderThenBody(t *testing.T) {
	r, _ := http.NewRequest("POST", "/v1/chat/completions", nil)
	if _, ok := callerPriority(r, map[string]any{}); ok {
		t.Fatal("priority from nothing")
	}
	if v, ok := callerPriority(r, map[string]any{"priority": float64(-5)}); !ok || v != -5 {
		t.Fatalf("body priority = %d %v", v, ok)
	}
	r.Header.Set("X-Priority", "3")
	if v, ok := callerPriority(r, map[string]any{"priority": float64(-5)}); !ok || v != 3 {
		t.Fatalf("header priority = %d %v, want header to win", v, ok)
	}
}

func TestMaxTokensOf(t *testing.T) {
	if n := maxTokensOf(map[string]any{"max_tokens": float64(900)}); n != 900 {
		t.Fatal(n)
	}
	if n := maxTokensOf(map[string]any{"max_completion_tokens": float64(50), "max_tokens": float64(900)}); n != 50 {
		t.Fatal(n)
	}
}

func TestTextBytesSkipsInlineMedia(t *testing.T) {
	body := map[string]any{"messages": []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": "hello"},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64," + string(make([]byte, 100000))}},
	}}}}
	if n := textBytesOf(body); n > 200 {
		t.Fatalf("text bytes %d counts the image", n)
	}
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
		rel, _, _, _, ok := q.Acquire(context.Background(), AdmissionRequest{Session: "chat", PromptTokens: 2000, Report: tr.rep})
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
	q, _ := newCacheQueue(t, func(c *AdmissionConfig) { c.Cache.Tokens = 640000 })
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
	if _, _, _, err := parseEnginePool(strings.NewReader("vllm:x 1\n")); err == nil {
		t.Fatal("want an error without cache_config_info")
	}
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 8, Cache: CacheConfig{Enabled: true}})
	if q.budget() != 0 {
		t.Fatal("pool should be unknown before SetPool")
	}
	q.SetPool(201, 3200, 64)
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
	for _, w := range q.Residency().Waiters {
		if w.Session == "c" && w.WaitS < 8*60 {
			t.Fatalf("retry waited %ds, want the first attempt's 8m30s", w.WaitS)
		}
	}
	mustWait(t, tr, "c retry")
}

// A running request's charge follows its streamed output past the expected output, and
// its overhead drops from the prefill to the decode figure once output streams.
func TestCacheAdmissionLiveCharge(t *testing.T) {
	q, _ := newCacheQueue(t, func(c *AdmissionConfig) {
		c.Cache.PrefillOverheadBlocks, c.Cache.DecodeOverheadBlocks = intPtr(28), intPtr(20)
	})
	gen := &atomic.Int64{}
	q.mu.Lock()
	w := &admissionWaiter{prompt: 100000, expOut: 4000, gen: gen}
	q.running[w] = struct{}{}
	base := q.runningLocked()
	gen.Store(20000)
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

// A thread that is back and waiting is never displaced by equal priority, however long
// others wait.
func TestCacheAdmissionWaitingOwnerKeepsHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	ta := q.start("a", 0, 200000)
	ra := mustAdmit(t, ta, "a")
	mustAdmit(t, q.start("b", 0, 280000), "b")
	ta.finish(ra, 200000, 3000)
	clk.add(10 * time.Minute)
	ta2 := q.start("a", 0, 260000) // back, but too big to run beside b
	tc := q.start("c", 0, 150000)  // would fit by taking a's hold
	clk.add(time.Hour)
	q.tick()
	mustWait(t, ta2, "a")
	mustWait(t, tc, "c, a's owner is waiting")
	for _, s := range q.Residency().Sessions {
		if s.Session == "a" && s.Held != 63*3200 {
			t.Fatalf("a's hold %d", s.Held)
		}
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
	q.mu.Lock()
	q.mirror = nil
	q.mu.Unlock()
	ta.finish(ra, 100000, 1000)
	mustAdmit(t, small, "small ahead of an older cold waiter")
	mustWait(t, cold, "cold behind small")
}

// The engine's usage counts prefixes it keeps between turns, so it is no floor on running.
func TestCacheAdmissionEngineUsageIsNoFloor(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	q.mu.Lock()
	defer q.mu.Unlock()
	q.engineUsed = 169 * 3200
	if got := q.runningLocked(); got != 0 {
		t.Fatalf("running %d blocks with nothing running", got/3200)
	}
}
