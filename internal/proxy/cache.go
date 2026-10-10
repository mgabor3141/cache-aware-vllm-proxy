package proxy

// Cache-aware admission. The engine's GPU prefix cache, not its sequence slots, is the
// scarce resource: a 200K-token agent thread out on a tool call holds no running request,
// so the engine evicts it first and the thread re-prefills everything when it returns.
//
// Inputs, all measured rather than guessed:
//   - every request is tokenized with the engine's /tokenize (exact size and token IDs);
//   - the KV event mirror (kvmirror.go) says how many of those tokens are cached now;
//   - prefill speed and its slowdown of decoding streams are measured constants;
//   - the chance an idle session returns is a measured curve of tool gaps.
//
// Space, in tokens against cache_tokens:
//   running request  its prompt plus expected output, in whole blocks. Never overcommitted.
//   idle session     its last prompt, held until displaced. Displacing it is all
//                    or nothing: with speculative decoding the engine drops a prompt's last
//                    cached block before matching, so losing the tail loses the whole hit.
//
// Order: priority (lower first), then a request whose prefix is cached, then small
// prompts, then cold ones, each by arrival.
//
// Releasing an idle session: a thread that got in cold keeps its cache for entryQuantum
// times its entry prefill. After that its hold is kept while its chance of returning per
// second, times its re-prefill (tokens / prefill speed, times the decoding it stalls),
// outweighs the pressure of the waiters it blocks (fitLocked). A waiter that outranks the
// holder displaces it at once. Nothing else forces a release: waiting long does not, so
// many waiters cannot start thrashing together.
//
// Holds steer admission only. The engine evicts free blocks least recently used first,
// whatever the proxy holds, so the mirror is checked each pass and a hold whose blocks
// are gone is dropped.
//
// Failure modes fall back conservatively: no mirror means nothing counts as cached; no
// /tokenize means a size estimate from text bytes.
// cache_tokens: 0 turns all of this off, leaving plain slot admission.

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

type CacheConfig struct {
	// Enabled turns on cache-aware admission; off is plain slot admission.
	Enabled bool `yaml:"enabled"`
	// Tokens is the engine's block pool in tokens. The proxy reads it from the engine's
	// /metrics (enginePool); this is only the value used before that (tests set it).
	Tokens int `yaml:"-"`
	// BlockTokens is the engine block size. The mirror reads it from the event stream;
	// this is only the value used before that (tests set it).
	BlockTokens int `yaml:"-"`

	// KVEventsEndpoint is the engine's KV event publisher; empty runs without a mirror.
	// Its replay service, which rebuilds the mirror on (re)subscribe, is on the next port.
	KVEventsEndpoint string `yaml:"kv_events_endpoint"`

	PrefillTokensPerSecond float64 `yaml:"prefill_tokens_per_s"`
	DecodeInterference     float64 `yaml:"decode_interference"`

	DefaultOutputTokens int `yaml:"default_output_tokens"`

	// Per running request, the blocks it takes beyond its prompt and output (recurrent
	// state, speculative and checkpoint blocks), while it prefills and once it decodes.
	// Engine and model specific: measure with `cavp -calibrate` on an idle engine. The
	// prefill figure is the peak; decode is lower. 0 is valid (a plain transformer).
	// Unset: 28 for both, the largest measured (a hybrid model on halt95's vLLM fork,
	// v2.2), and decode falls back to the prefill figure. Too low a figure is caught by the
	// overcommit guard (guard.go).
	PrefillOverheadBlocks *int `yaml:"prefill_overhead_blocks"`
	DecodeOverheadBlocks  *int `yaml:"decode_overhead_blocks"`
}

const defaultOverheadBlocks = 28

func (c CacheConfig) overhead(decoding bool) int {
	if decoding && c.DecodeOverheadBlocks != nil {
		return *c.DecodeOverheadBlocks
	}
	if c.PrefillOverheadBlocks != nil {
		return *c.PrefillOverheadBlocks
	}
	return defaultOverheadBlocks
}

const (
	// tokenizeTimeout bounds /tokenize including the wait for a tokenize slot.
	tokenizeTimeout = 10 * time.Second
	// Caller-set priorities are clamped to this range.
	minCallerPriority, maxCallerPriority = -10, 100
	// fallbackBytesPerToken prices a prompt /tokenize could not measure.
	fallbackBytesPerToken = 4
)

// tokensPerChunk bounds output tokens per streamed chunk: one plus the MTP draft length
// (K=3). Counting chunks times this over-counts by at most a block per request.
const tokensPerChunk = 4

// retryMemory: a session's request that arrives within this long after its previous
// waiter's client gave up inherits that waiter's start time.
const retryMemory = 5 * time.Minute

// minHold: an idle thread keeps its cache at least this long, so the quick tool calls
// (file reads, greps) always return to a warm cache. Measured 2026-10-08: of 20 review
// agents, 91% of returns came within 1 s and 100% within 5 s; over 72 h of 50K+ turns,
// 58% within 5 s.
const minHold = 5 * time.Second

// smallPrompt: requests below this are cheap to run and hold little, and go ahead of
// cold work.
const smallPrompt = 16000

// evictBelowRunning: a cold request may displace an expendable idle hold only while
// fewer requests than this run. With more running, the engine is busy anyway and the
// displacement would only add a re-prefill.
const evictBelowRunning = 4

func (c *CacheConfig) applyDefaults() {
	if c.BlockTokens <= 0 {
		c.BlockTokens = 1
	}

	if c.PrefillTokensPerSecond <= 0 {
		c.PrefillTokensPerSecond = 4200
	}
	if c.DecodeInterference < 0 {
		c.DecodeInterference = 0
	}

	if c.DefaultOutputTokens <= 0 {
		c.DefaultOutputTokens = 2048
	}

}

func (c CacheConfig) validate() error {
	if c.DecodeInterference > 1 {
		return fmt.Errorf("admission.cache.decode_interference %g: want 0-1 (the share of decode speed a prefill takes)", c.DecodeInterference)
	}
	for name, v := range map[string]*int{"prefill_overhead_blocks": c.PrefillOverheadBlocks, "decode_overhead_blocks": c.DecodeOverheadBlocks} {
		if v != nil && *v < 0 {
			return fmt.Errorf("admission.cache.%s %d: must not be negative", name, *v)
		}
	}
	return nil
}

func (c CacheConfig) enabled() bool { return c.Enabled }

// block is the engine block size: from the mirror once known, else the configured fallback.
func (q *AdmissionQueue) block() int {
	if q.mirror != nil {
		if b := q.mirror.BlockSize(); b > 0 {
			return b
		}
	}
	return q.cfg.Cache.BlockTokens
}

// budget is the engine's block pool in tokens; 0 until known, which means slot admission.
func (q *AdmissionQueue) budget() int { return q.roundDown(q.pool) }

// SetPool records the engine's block pool (vllm:cache_config_info num_gpu_blocks x
// block_size). One block is the engine's reserved null block.
func (q *AdmissionQueue) SetPool(blocks, blockSize, unit int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.mirror != nil {
		q.mirror.SetGeometry(blockSize, unit)
	}
	q.pool = max(blocks-1, 0) * blockSize
	q.cfg.Cache.BlockTokens = blockSize // until the mirror has seen an event
	q.grantLocked()
}

func (q *AdmissionQueue) roundUp(tokens int) int {
	b := q.block()
	return (tokens + b - 1) / b * b
}

func (q *AdmissionQueue) roundDown(tokens int) int { return tokens / q.block() * q.block() }

func (c CacheConfig) prefillSeconds(tokens int) float64 {
	return float64(max(tokens, 0)) / c.PrefillTokensPerSecond
}

const (
	sessionSamples   = 16
	sessionRetention = 24 * time.Hour
	sessionCap       = 4096
)

type cacheSession struct {
	priority   int
	footprint  int // prompt + completion of the last completed turn
	lastPrompt int // prompt tokens of the last completed turn
	outputs    []int

	running int
	// Largest observation of the current busy period (overlapping requests).
	busyFootprint, busyPrompt int
	busyBlocks                kvmirror.PromptKeys
	lastDone                  time.Time
	held                      int                 // prompt tokens protected while idle (whole blocks); 0 when not holding
	heldBlocks                kvmirror.PromptKeys // mirror keys of the held prompt
	heldVisible               bool                // the mirror saw the hold cached when it began
	heldEpoch                 uint64              // mirror subscription it was seen under

	gaveUpSince, gaveUpAt time.Time // a waiter whose client left: since when it waited, and when it left
	waiting               int       // requests of this session in the queue
	lastSeen              time.Time
}

func pushInt(s []int, v int) []int {
	s = append(s, v)
	if len(s) > sessionSamples {
		s = s[len(s)-sessionSamples:]
	}
	return s
}

func percentileInt(s []int, p float64) int {
	c := append([]int(nil), s...)
	sort.Ints(c)
	return c[int(p*float64(len(c)-1)+0.5)]
}

// holding: idle with its prompt kept. Holds do not expire; under pressure they are
// released by the return-rate rule in fitLocked, without pressure they cost nothing.
func (s *cacheSession) holding(now time.Time) bool {
	return s.running == 0 && s.held > 0
}

// expectedOutput is the output reserved for a request before it streams: the session's
// p75, or default_output_tokens without history. Output beyond it is charged as it
// streams (liveCharge), so this only has to be typical, not worst case.
func (q *AdmissionQueue) expectedOutput(s *cacheSession, maxTokens int) int {
	out := q.cfg.Cache.DefaultOutputTokens
	if s != nil && len(s.outputs) > 0 {
		out = percentileInt(s.outputs, 0.75)
	}
	if maxTokens > 0 {
		out = min(out, maxTokens)
	}
	return out
}

// FallbackPrompt estimates prompt tokens when /tokenize failed or was skipped.
func FallbackPrompt(textBytes int64) int { return int(textBytes / fallbackBytesPerToken) }

func (q *AdmissionQueue) session(key string, now time.Time) *cacheSession {
	if key == "" {
		return nil
	}
	s := q.sessions[key]
	if s == nil {
		if len(q.sessions) >= sessionCap {
			q.pruneSessionsLocked(now, true)
		}
		s = &cacheSession{}
		q.sessions[key] = s
	}
	s.lastSeen = now
	return s
}

func (q *AdmissionQueue) pruneSessionsLocked(now time.Time, force bool) {
	var oldestKey string
	var oldest time.Time
	for k, s := range q.sessions {
		if s.running > 0 {
			continue
		}
		if now.Sub(s.lastSeen) > sessionRetention {
			delete(q.sessions, k)
			continue
		}
		if oldestKey == "" || s.lastSeen.Before(oldest) {
			oldestKey, oldest = k, s.lastSeen
		}
	}
	if force && len(q.sessions) >= sessionCap && oldestKey != "" {
		delete(q.sessions, oldestKey)
	}
}

func (q *AdmissionQueue) heldLocked(now time.Time) int {
	held := 0
	for _, s := range q.sessions {
		if s.holding(now) {
			held += s.held
		}
	}
	return held
}

// generated is a request's output so far (0 before it streams or when not streaming).
func (w *admissionWaiter) generated() int {
	if w.gen == nil {
		return 0
	}
	return int(w.gen.Load())
}

// liveCharge is what a request occupies or will soon: its prompt and the larger of its
// output so far and its expected output, in whole blocks, plus its recurrent state.
func (q *AdmissionQueue) liveCharge(w *admissionWaiter) int {
	g := w.generated()
	return q.roundUp(w.prompt+max(g, w.expOut)) + q.overheadLocked(g > 0)*q.block()
}

// runningLocked is the cache running requests occupy: the sum of their live charges.
// The engine's kv_cache_usage_perc is not a floor: since v2.5.1 it also counts finished
// turns' prefixes the engine keeps referenced until it needs the room (grace holds, which
// yield before any preemption), so it reads high with nothing running. It stays in
// /debug/residency for comparison, and a preemption is logged (runEngineUsage).
func (q *AdmissionQueue) runningLocked() int {
	model := 0
	for w := range q.running {
		model += q.liveCharge(w)
	}
	return model
}

// modelActualLocked is the model's estimate of what running requests occupy now
// (output so far, no reserve), to compare with the engine's count.
func (q *AdmissionQueue) modelActualLocked() int {
	n := 0
	for w := range q.running {
		g := w.generated()
		n += q.roundUp(w.prompt+g) + q.overheadLocked(g > 0)*q.block()
	}
	return n
}

// SetEngineUsed records the engine's occupied blocks (vllm:kv_cache_usage_perc of the
// pool): blocks referenced by running requests. Cached blocks nothing uses count as free.
func (q *AdmissionQueue) SetEngineUsed(tokens int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.engineUsed = tokens
	q.grantLocked()
}

func (q *AdmissionQueue) cacheUsedLocked(now time.Time) int {
	return q.runningLocked() + q.heldLocked(now)
}

// cachedLocked is the waiter's prefix the engine holds now, per the mirror.
func (q *AdmissionQueue) cachedLocked(w *admissionWaiter) int {
	if q.mirror == nil || w.blocks.Empty() || !q.mirror.Healthy() {
		return 0
	}
	return min(q.mirror.Cached(w.blocks), w.prompt)
}

// ownCost is the GPU seconds a waiter's prefill takes, counting the decoding streams it
// stalls while it runs.
func (q *AdmissionQueue) ownCost(w *admissionWaiter, cached int) float64 {
	return q.cfg.Cache.prefillSeconds(w.prompt-cached) * (1 + q.cfg.Cache.DecodeInterference*float64(q.active))
}

type displacement struct {
	s      *cacheSession
	tokens int
}

// needLocked is how many held tokens w would have to displace to run now; <= 0 when it
// fits beside running work and holds (including the small hold overcommit).
func (q *AdmissionQueue) needLocked(w *admissionWaiter, now time.Time) int {
	budget := q.budget()
	charge := min(w.charge, budget)
	own := q.sessions[w.session]
	free := budget - q.cacheUsedLocked(now)
	if own != nil && own.holding(now) {
		free += own.held
	}
	return charge - free
}

// holdFor is how long an idle hold is firm: as long as its re-prefill would take, with
// the decoding it would stall (ski rental: hold for what rebuying costs), and at least
// minHold. Past it the hold is expendable: kept until someone needs the room.
func (q *AdmissionQueue) holdFor(s *cacheSession) time.Duration {
	cc := q.cfg.Cache
	sec := cc.prefillSeconds(s.held) * (1 + cc.DecodeInterference*float64(q.active))
	return max(minHold, time.Duration(sec*float64(time.Second)))
}

// firm: an idle hold no equal or lower priority waiter may displace, because its
// thread is back and waiting, or is likely back soon.
func (q *AdmissionQueue) firm(s *cacheSession, now time.Time) bool {
	return s.waiting > 0 || now.Sub(s.lastDone) < q.holdFor(s)
}

// tier orders waiters within a priority: a thread whose prefix is cached first (nothing
// may evict it before it runs), then small requests, then cold work.
func (q *AdmissionQueue) tier(w *admissionWaiter, cached int) int {
	own := q.sessions[w.session]
	switch {
	case cached > 0 && cached*2 >= w.prompt,
		own != nil && own.holding(q.now()) && own.held*2 >= w.prompt:
		return 0
	case w.prompt < smallPrompt:
		return 1
	}
	return 2
}

// fitLocked decides whether w can run now within free room (budget minus running work,
// holds, and what earlier blocked waiters reserved), and which idle holds it displaces.
// A higher priority may displace any lower-priority idle hold. Otherwise only expendable
// holds (past holdFor, owner not waiting) go, and only while the engine is under-used.
func (q *AdmissionQueue) fitLocked(w *admissionWaiter, now time.Time, reserved int) (plan []displacement, ok bool) {
	need := q.needLocked(w, now) + reserved
	if need <= 0 {
		return nil, true
	}
	var cands []*cacheSession
	for k, s := range q.sessions {
		if k == w.session || !s.holding(now) {
			continue
		}
		// With nothing running, the head waiter may take any equal-priority hold: nothing
		// else will free room, so this guarantees progress.
		if w.priority < s.priority || (q.active == 0 && w.priority == s.priority) ||
			(!q.firm(s, now) && q.active < evictBelowRunning) {
			cands = append(cands, s)
		}
	}
	// Best fit: the smallest single hold that covers the need; otherwise the holds idle
	// longest (least likely back) until it is covered.
	var single *cacheSession
	for _, s := range cands {
		if s.held >= need && (single == nil || s.held < single.held) {
			single = s
		}
	}
	if single != nil {
		return []displacement{{single, single.held}}, true
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].lastDone.Before(cands[j].lastDone) })
	for _, s := range cands {
		if need <= 0 {
			break
		}
		plan = append(plan, displacement{s, s.held})
		need -= s.held
	}
	if need > 0 {
		return nil, false
	}
	return plan, true
}

// grantCacheLocked admits waiters in order: priority, then tier, then longest wait. All
// queueing happens here; the engine is never sent more than it can run. A waiter that
// does not fit reserves the room it still needs, so later waiters use only what is left
// beyond it and cannot starve it.
func (q *AdmissionQueue) grantCacheLocked() {
	now := q.now()
	q.syncHoldsLocked(now)
	type scored struct {
		w      *admissionWaiter
		cached int
		tier   int
	}
	order := make([]scored, 0, len(q.waiters))
	for _, w := range q.waiters {
		cached := q.cachedLocked(w)
		order = append(order, scored{w, cached, q.tier(w, cached)})
	}
	sort.SliceStable(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if a.w.priority != b.w.priority {
			return a.w.priority < b.w.priority
		}
		if a.tier != b.tier {
			return a.tier < b.tier
		}
		if !a.w.enqueued.Equal(b.w.enqueued) {
			return a.w.enqueued.Before(b.w.enqueued)
		}
		return a.w.sequence < b.w.sequence
	})
	reserved := 0
	for _, o := range order {
		if q.active >= q.cfg.MaxActive {
			return
		}
		w := o.w
		plan, ok := q.fitLocked(w, now, reserved)
		if !ok {
			reserved += max(q.needLocked(w, now), 0)
			continue
		}
		displaced, cost := 0, 0.0
		for _, d := range plan {
			cost += q.cfg.Cache.prefillSeconds(d.tokens)
			q.dropHoldLocked(d.s)
			displaced += d.tokens
			q.displaced += d.tokens
		}
		if w.report != nil {
			w.report.PredictedCached = o.cached
			w.report.PrefillCostS = q.ownCost(w, o.cached)
			w.report.DisplacedTokens = displaced
			w.report.DisplaceCostS = cost
		}
		q.removeWaiterLocked(w)
		q.admitLocked(w, now)
		close(w.ready)
	}
}

func (q *AdmissionQueue) dropHoldLocked(s *cacheSession) {
	s.held = 0
	s.heldBlocks = kvmirror.PromptKeys{}
	s.heldVisible = false
}

// syncHoldsLocked drops holds the engine evicted. Only on evidence: the mirror must have
// seen most of the hold cached when it began (heldVisible), and then see it gone. Holds
// it never saw (image blocks carry extra hash keys; threads older than a resubscribe's
// replay window) are unknown, not evicted.
func (q *AdmissionQueue) syncHoldsLocked(now time.Time) {
	if q.mirror == nil || !q.mirror.Healthy() {
		return
	}
	for _, s := range q.sessions {
		if !s.holding(now) || !s.heldVisible || s.heldEpoch != q.mirror.Epoch() {
			continue // never seen, or seen before a resubscribe whose replay may not reach it
		}
		if q.mirror.Cached(s.heldBlocks) < s.held/2 {
			if q.firm(s, now) {
				q.guardEvictionLocked(now, s.held)
			}
			q.dropHoldLocked(s)
			q.evictedHolds++
		}
	}
}

func (q *AdmissionQueue) admitLocked(w *admissionWaiter, now time.Time) {
	q.active++
	w.granted = true
	if !q.cfg.Cache.enabled() {
		return
	}
	q.running[w] = struct{}{}
	if s := q.sessions[w.session]; s != nil {
		s.running++
		s.held = 0
		s.heldBlocks = kvmirror.PromptKeys{}
		s.heldVisible = false
		s.priority = w.priority
	}
}

// finishLocked returns a request's charge and records what the engine reported. While
// a session has overlapping requests, the largest observation wins: a side call does not
// shrink what the main thread left cached. A busy period with no usage at all (upstream
// error, client gone) leaves no hold, since nothing says anything was cached.
func (q *AdmissionQueue) finishLocked(w *admissionWaiter, obs func() (CacheObservation, bool)) {
	if !q.cfg.Cache.enabled() {
		return
	}
	now := q.now()
	delete(q.running, w)
	s := q.sessions[w.session]
	if s == nil {
		return
	}
	if o, ok := obs(); ok && o.PromptTokens > 0 {
		if fp := o.PromptTokens + o.CompletionTokens; fp > s.busyFootprint {
			s.busyFootprint = fp
			s.busyPrompt = o.PromptTokens

			s.busyBlocks = w.blocks
		}
		s.outputs = pushInt(s.outputs, o.CompletionTokens)
	}
	if s.running > 0 {
		s.running--
	}
	if s.running > 0 {
		return
	}
	s.lastDone = now
	s.held, s.heldBlocks = 0, kvmirror.PromptKeys{}
	if s.busyFootprint > 0 {
		s.footprint, s.lastPrompt = s.busyFootprint, s.busyPrompt
		// The next turn resends the prompt, not the completion (thinking is dropped), so
		// the prompt is what stays useful in the cache.
		s.held = q.roundUp(s.lastPrompt) // with tail hits its partial last block is kept too
		s.heldBlocks = s.busyBlocks
		s.heldVisible = q.mirror != nil && q.mirror.Healthy() && !s.heldBlocks.Empty() &&
			q.mirror.Cached(s.heldBlocks) >= s.held/2
		if q.mirror != nil {
			s.heldEpoch = q.mirror.Epoch()
		}
	}
	s.busyFootprint, s.busyPrompt, s.busyBlocks = 0, 0, kvmirror.PromptKeys{}
}

// CacheObservation is the engine's usage for one finished request.
type CacheObservation struct {
	PromptTokens     int
	CompletionTokens int
}

type ResidencySession struct {
	Session   string `json:"session"`
	Priority  int    `json:"priority"`
	Footprint int    `json:"footprint_tokens"`
	Held      int    `json:"held_tokens"`
	Running   int    `json:"running"`
	IdleS     int    `json:"idle_s,omitempty"`
	FirmS     int    `json:"firm_s,omitempty"` // seconds left before the hold is expendable
}

type ResidencyWaiter struct {
	Session  string  `json:"session,omitempty"`
	Priority int     `json:"priority"`
	WaitS    int     `json:"wait_s"`
	Prompt   int     `json:"prompt_tokens"`
	Cached   int     `json:"cached_tokens"`
	Charge   int     `json:"charge_tokens"`
	PrefillS float64 `json:"prefill_cost_s"`
}

type ResidencySnapshot struct {
	Enabled       bool `json:"enabled"`
	BudgetTokens  int  `json:"budget_tokens"`
	UsedTokens    int  `json:"used_tokens"`
	RunningTokens int  `json:"running_tokens"`
	HeldTokens    int  `json:"held_tokens"`
	// The engine's count of running requests' blocks against the model's estimate of the
	// same (output so far, no reserve): their gap is the accounting error.
	EngineUsedTokens  int                `json:"engine_used_tokens"`
	ModelActualTokens int                `json:"model_actual_tokens"`
	Active            int                `json:"active"`
	Queued            int                `json:"queued"`
	Displaced         int                `json:"displaced_tokens_total"`
	EvictedHolds      int                `json:"holds_evicted_by_engine_total"`
	MarginBlocks      int                `json:"guard_margin_blocks"`
	MarginRaises      int                `json:"guard_raises_total"`
	Mirror            *kvmirror.Status   `json:"mirror,omitempty"`
	Sessions          []ResidencySession `json:"resident"`
	Waiters           []ResidencyWaiter  `json:"waiting"`
}

func (q *AdmissionQueue) Residency() ResidencySnapshot {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := q.now()
	cc := q.cfg.Cache
	out := ResidencySnapshot{
		Enabled: cc.enabled(), BudgetTokens: q.budget(), UsedTokens: q.cacheUsedLocked(now),
		RunningTokens: q.runningLocked(), HeldTokens: q.heldLocked(now),
		EngineUsedTokens: q.engineUsed, ModelActualTokens: q.modelActualLocked(),
		Active: q.active, Queued: len(q.waiters), Displaced: q.displaced, EvictedHolds: q.evictedHolds,
		MarginBlocks: q.margin, MarginRaises: q.marginRaises,
	}
	if q.mirror != nil {
		st := q.mirror.Status()
		out.Mirror = &st
	}
	for k, s := range q.sessions {
		if s.running == 0 && !s.holding(now) {
			continue
		}
		rs := ResidencySession{Session: k, Priority: s.priority, Footprint: s.footprint, Running: s.running}
		if s.holding(now) {
			rs.Held = s.held
			rs.IdleS = int(now.Sub(s.lastDone).Seconds())
			rs.FirmS = max(0, int((q.holdFor(s) - now.Sub(s.lastDone)).Seconds()))
		}
		out.Sessions = append(out.Sessions, rs)
	}
	sort.Slice(out.Sessions, func(i, j int) bool { return out.Sessions[i].Footprint > out.Sessions[j].Footprint })
	for _, w := range q.waiters {
		cached := q.cachedLocked(w)
		out.Waiters = append(out.Waiters, ResidencyWaiter{
			Session: w.session, Priority: w.priority, WaitS: int(now.Sub(w.enqueued).Seconds()),
			Prompt: w.prompt, Cached: cached, Charge: w.charge,
			PrefillS: math.Round(q.ownCost(w, cached)*10) / 10,
		})
	}
	return out
}
