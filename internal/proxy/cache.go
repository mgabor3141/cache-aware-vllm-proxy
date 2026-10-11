package proxy

// Cache-aware admission. The engine's GPU prefix cache, not its sequence slots, is the
// scarce resource: a 200K-token agent thread out on a tool call holds no running request,
// so the engine evicts it first and the thread re-prefills everything when it returns.
//
// Inputs, all measured rather than guessed:
//   - every request is tokenized with the engine's /tokenize (exact size and token IDs);
//   - the KV event mirror (kvmirror) says how many of those tokens are cached now;
//   - the engine's KV pool and block size come from its /metrics (enginepool.go);
//   - prefill speed and its slowdown of decoding streams are measured constants.
//
// Space, in tokens against the engine's pool:
//   running request  its prompt plus the larger of its output so far and its expected
//                    output, in whole blocks, plus a per-request overhead (guard.go).
//   session hold     its last prompt, held until displaced, and still counted while
//                    its own next request runs (the larger of the two). Displacing it is all
//                    or nothing: with speculative decoding the engine drops a prompt's last
//                    cached block before matching, so losing the tail loses the whole hit.
//
// Order: priority (lower first), then a request whose prefix is cached, then small
// prompts, then cold ones, each by arrival.
//
// Releasing an idle session: its hold is firm for as long as re-prefilling it would take
// (with the decoding that prefill stalls), at least minHold, and as long again once its own thread
// starts waiting (holdFor, firm). After that it is expendable: an equal-priority waiter may
// displace it while fewer than evictBelowRunning requests run. A higher-priority waiter
// may displace any lower-priority hold, idle or not (displaceable). Waiting long forces nothing, so many
// waiters cannot start thrashing together.
//
// Holds steer admission only. The engine evicts free blocks least recently used first,
// whatever the proxy holds, so the mirror is checked each pass and a hold whose blocks
// are gone is dropped.
//
// Until the pool is known, nothing is admitted. While the mirror resyncs, nothing counts
// as cached; without /tokenize the prompt size is estimated from text bytes.

import (
	"fmt"
	"log"
	"sort"
	"sync/atomic"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

type CacheConfig struct {
	// KVEventsEndpoint is the engine's KV event publisher (required). Its replay
	// service, which rebuilds the mirror on (re)subscribe, is on the next port.
	KVEventsEndpoint string `yaml:"kv_events_endpoint"`

	// SpeculativeDecoding says the engine runs speculative decoding (EAGLE, MTP:
	// --speculative-config). vLLM then serves a cached prefix minus its last block (or
	// unit), and the mirror predicts the same. Default false (plain vLLM).
	SpeculativeDecoding bool `yaml:"speculative_decoding"`

	PrefillTokensPerSecond float64 `yaml:"prefill_tokens_per_s"`
	DecodeInterference     float64 `yaml:"decode_interference"`

	DefaultOutputTokens int `yaml:"default_output_tokens"`

	// TokensPerChunk is how many output tokens a streamed chunk is counted as when the
	// stream carries no per-chunk usage: 1 without speculative decoding, up to 1 plus the
	// draft length with it (4 for MTP with 3 draft tokens). Default 1.
	TokensPerChunk int `yaml:"tokens_per_chunk"`

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

// OutputCounter counts a streaming request's output tokens so far for admission. The
// request path feeds it each streamed chunk: from the chunk's usage when the stream
// carries one per chunk (stream_options.continuous_usage_stats), else tokens_per_chunk per
// choice chunk.
type OutputCounter struct {
	n        atomic.Int64
	perChunk int64
}

// NewOutputCounter returns a counter for one request.
func (q *AdmissionQueue) NewOutputCounter() *OutputCounter {
	return &OutputCounter{perChunk: int64(q.cfg.Cache.TokensPerChunk)}
}

// Chunk counts one streamed choice chunk that carried no usage.
func (c *OutputCounter) Chunk() { c.n.Add(c.perChunk) }

// Usage records the completion tokens a chunk's usage reports (a running total).
func (c *OutputCounter) Usage(completionTokens int64) { c.n.Store(completionTokens) }

// Load is the output so far.
func (c *OutputCounter) Load() int64 { return c.n.Load() }

// retryMemory: a session's request that arrives within this long after its previous
// waiter's client gave up inherits that waiter's arrival time (one attempt's, never a chain).
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
	if c.PrefillTokensPerSecond <= 0 {
		c.PrefillTokensPerSecond = 4200
	}
	if c.DecodeInterference < 0 {
		c.DecodeInterference = 0
	}

	if c.DefaultOutputTokens <= 0 {
		c.DefaultOutputTokens = 2048
	}
	if c.TokensPerChunk <= 0 {
		c.TokensPerChunk = 1
	}
}

func (c CacheConfig) validate() error {
	if c.KVEventsEndpoint == "" {
		return fmt.Errorf("admission.cache.kv_events_endpoint is required (the engine's --kv-events-config ZMQ publisher)")
	}
	if kvmirror.ReplayEndpoint(c.KVEventsEndpoint) == "" {
		return fmt.Errorf("admission.cache.kv_events_endpoint %q: want host:port (the replay service is on the next port)", c.KVEventsEndpoint)
	}
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

// block is the engine block size; 0 until known.
func (q *AdmissionQueue) block() int { return q.blockSize }

// budget is the engine's block pool in tokens; 0 until known, and nothing is admitted until then.
func (q *AdmissionQueue) budget() int { return q.roundDown(q.pool) }

// SetPool records the engine's block pool (vllm:cache_config_info num_gpu_blocks x
// block_size). One block is the engine's reserved null block. unit is its prefix_match_unit
// (0: none, or not shown), which the mirror needs to match unit hits.
func (q *AdmissionQueue) SetPool(blocks, blockSize, unit int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.mirror.SetGeometry(blockSize, unit)
	q.pool = max(blocks-1, 0) * blockSize
	q.blockSize = blockSize
	q.grantLocked()
}

func (q *AdmissionQueue) roundUp(tokens int) int {
	b := max(q.block(), 1)
	return (tokens + b - 1) / b * b
}

func (q *AdmissionQueue) roundDown(tokens int) int { b := max(q.block(), 1); return tokens / b * b }

func (c CacheConfig) prefillSeconds(tokens int) float64 {
	return float64(max(tokens, 0)) / c.PrefillTokensPerSecond
}

const (
	sessionSamples   = 16
	sessionRetention = 24 * time.Hour
	sessionCap       = 4096
)

type cacheSession struct {
	key      string
	priority int // of its hold: the request that left it
	outputs  []int

	running int
	// Largest observation of the current busy period (overlapping requests).
	busyFootprint, busyPrompt, busyPriority int
	busyBlocks                              kvmirror.PromptKeys
	lastDone                                time.Time
	held                                    int                 // prompt tokens protected while idle (whole blocks); 0 when not holding
	heldBlocks                              kvmirror.PromptKeys // mirror keys of the held prompt
	// heldSeen is what the mirror saw cached of the hold: -1 until the mirror is healthy to
	// ask, 0 when it saw nothing (the hold cannot be verified).
	heldSeen int
	heldGen  uint64 // the mirror's generation when the turn that left the hold was admitted
	busyGen  uint64

	gaveUpSince, gaveUpAt time.Time // a waiter whose client left: since when it waited, and when it left
	waiting               int       // requests of this session in the queue
	waitingSince          time.Time // when waiting last went from 0 to 1
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

func (q *AdmissionQueue) session(key string, now time.Time) *cacheSession {
	if key == "" {
		return nil
	}
	s := q.sessions[key]
	if s == nil {
		if len(q.sessions) >= sessionCap {
			q.pruneSessionsLocked(now, true)
		}
		s = &cacheSession{key: key}
		q.sessions[key] = s
	}
	s.lastSeen = now
	return s
}

// pruneSessionsLocked forgets sessions idle past sessionRetention. With force (the table
// is full), it also forgets the least recently seen idle sessions, those holding nothing
// first, down to sessionCap less a sixteenth, so a full table is not scanned for every
// new session. A session with running or queued requests is never forgotten; a hold
// forgotten that way counts as displaced, and is logged.
func (q *AdmissionQueue) pruneSessionsLocked(now time.Time, force bool) {
	var idle []string
	for k, s := range q.sessions {
		if s.running > 0 || s.waiting > 0 {
			continue
		}
		if now.Sub(s.lastSeen) > sessionRetention {
			delete(q.sessions, k)
		} else if force {
			idle = append(idle, k)
		}
	}
	excess := len(q.sessions) - (sessionCap - sessionCap/16)
	if !force || excess <= 0 {
		return
	}
	sort.Slice(idle, func(i, j int) bool {
		a, b := q.sessions[idle[i]], q.sessions[idle[j]]
		if (a.held > 0) != (b.held > 0) {
			return a.held == 0
		}
		if !a.lastSeen.Equal(b.lastSeen) {
			return a.lastSeen.Before(b.lastSeen)
		}
		return idle[i] < idle[j]
	})
	forgotten, held := 0, 0
	for _, k := range idle[:min(excess, len(idle))] {
		if h := q.sessions[k].held; h > 0 {
			forgotten++
			held += h
		}
		delete(q.sessions, k)
	}
	if forgotten > 0 {
		q.displaced += held
		log.Printf("level=warn msg=\"session table full: forgot idle holds\" holds=%d tokens=%d sessions=%d", forgotten, held, len(q.sessions))
	}
}

// heldLocked is every session's held prompt, idle or not: a session's own running request
// does not release the prefix the engine still has cached.
func (q *AdmissionQueue) heldLocked(now time.Time) int {
	held := 0
	for _, s := range q.sessions {
		held += s.held
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

// chargeLocked is what a waiter would occupy if admitted now: priced at the current block
// size, guard margin and session history, not frozen at arrival.
func (q *AdmissionQueue) chargeLocked(w *admissionWaiter) int {
	if w.buffer {
		w.expOut = w.maxOut
		if w.expOut <= 0 {
			w.expOut = q.cfg.Cache.DefaultOutputTokens
		}
	} else {
		w.expOut = q.expectedOutput(q.sessions[w.session], w.maxOut)
	}
	return q.liveCharge(w)
}

// runningLocked is the cache running requests occupy: the sum of their live charges.
// The engine's kv_cache_usage_perc is no measure of this: it also counts finished turns'
// prefixes an engine may keep referenced until it needs the room, so it reads high with
// nothing running. The proxy does not read it.
func (q *AdmissionQueue) runningLocked() int {
	model := 0
	for w := range q.running {
		model += q.liveCharge(w)
	}
	return model
}

// cacheUsedLocked is the modelled occupancy: running requests and holds, a session with
// both charged the larger of its hold and its running requests (they share its prefix).
func (q *AdmissionQueue) cacheUsedLocked(now time.Time) int {
	used := q.heldLocked(now)
	own := map[*cacheSession]int{}
	for w := range q.running {
		if s := q.sessions[w.session]; s != nil && s.held > 0 {
			own[s] += q.liveCharge(w)
		} else {
			used += q.liveCharge(w)
		}
	}
	for s, run := range own {
		used += max(run, s.held) - s.held
	}
	return used
}

// sessionRunningLocked is the live charge of a session's running requests.
func (q *AdmissionQueue) sessionRunningLocked(s *cacheSession) int {
	run := 0
	for w := range q.running {
		if q.sessions[w.session] == s {
			run += q.liveCharge(w)
		}
	}
	return run
}

// cachedLocked is the waiter's prefix the engine holds now, per the mirror.
func (q *AdmissionQueue) cachedLocked(w *admissionWaiter) int {
	if w.blocks.Empty() || !q.mirror.Healthy() {
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

// addsLocked is the room w would add to the modelled use if admitted now: its charge, of
// which its session's hold already covers part, clamped to the budget so a request larger
// than the pool can still run alone.
func (q *AdmissionQueue) addsLocked(w *admissionWaiter) int {
	charge := min(q.chargeLocked(w), q.budget())
	if own := q.sessions[w.session]; own != nil && own.held > 0 {
		// The session is charged the larger of its hold and its running requests.
		run := q.sessionRunningLocked(own)
		charge = max(own.held, run+charge) - max(own.held, run)
	}
	return charge
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
// thread is likely back soon, or is back and waiting. Either lasts holdFor: from the turn
// that left the hold, or from when the thread started waiting, so a waiter that cannot
// run does not pin its hold.
func (q *AdmissionQueue) firm(s *cacheSession, now time.Time) bool {
	hold := q.holdFor(s)
	return now.Sub(s.lastDone) < hold || (s.waiting > 0 && now.Sub(s.waitingSince) < hold)
}

// tier orders waiters within a priority: a thread whose prefix is cached first (nothing
// may evict it before it runs), then small requests, then cold work.
func (q *AdmissionQueue) tier(w *admissionWaiter, cached int, now time.Time) int {
	own := q.sessions[w.session]
	switch {
	case cached > 0 && cached*2 >= w.prompt,
		own != nil && own.holding(now) && own.held*2 >= w.prompt:
		return 0
	case w.prompt < smallPrompt:
		return 1
	}
	return 2
}

// grantState is what a grant pass knows of the queue. It is rebuilt after each admission
// (which changes it), so a pass costs a scan of the sessions per admission, not per waiter.
type grantState struct {
	free      int         // budget less modelled use
	reach     map[int]int // by waiter priority: held tokens a waiter may displace
	heldAbove map[int]int // held tokens by hold priority
}

func (q *AdmissionQueue) grantStateLocked(now time.Time) *grantState {
	g := &grantState{free: q.budget() - q.cacheUsedLocked(now), reach: map[int]int{}, heldAbove: map[int]int{}}
	for _, s := range q.sessions {
		g.heldAbove[s.priority] += s.held
	}
	return g
}

// displaceable: a waiter at priority p may displace s. A higher priority may displace any
// lower-priority hold, even while its session runs a request (the running request stays
// charged; only the hold goes), so an agent that overlaps calls cannot pin its hold.
// Otherwise only idle, expendable holds (past holdFor) go, and only while the engine is
// under-used. With nothing running, the head waiter may take any equal-priority hold:
// nothing else will free room, so this guarantees progress.
func (q *AdmissionQueue) displaceable(p int, s *cacheSession, now time.Time) bool {
	if s.held > 0 && p < s.priority {
		return true
	}
	return s.holding(now) && ((q.active == 0 && p == s.priority) ||
		(!q.firm(s, now) && q.active < evictBelowRunning))
}

// reachLocked is how many held tokens w may displace.
func (q *AdmissionQueue) reachLocked(g *grantState, w *admissionWaiter, now time.Time) int {
	r, ok := g.reach[w.priority]
	if !ok {
		for _, s := range q.sessions {
			if q.displaceable(w.priority, s, now) {
				r += s.held
			}
		}
		g.reach[w.priority] = r
	}
	if own := q.sessions[w.session]; own != nil && q.displaceable(w.priority, own, now) {
		r -= own.held
	}
	return r
}

// fitLocked decides whether w can run now within free room (budget minus running work,
// holds, and what earlier blocked waiters reserved), and which idle holds it displaces.
func (q *AdmissionQueue) fitLocked(w *admissionWaiter, now time.Time, reserved int, g *grantState) (plan []displacement, ok bool) {
	// A waiter its session's hold already covers (it resends a prefix of the held prompt
	// and fits inside it) takes no room: reservations do not hold it back.
	adds := q.addsLocked(w)
	if adds <= 0 && q.sharesHold(w) {
		return nil, true
	}
	need := adds - g.free + reserved
	if need <= 0 {
		return nil, true
	}
	if need > q.reachLocked(g, w, now) {
		return nil, false
	}
	var cands []*cacheSession
	for k, s := range q.sessions {
		if k != w.session && q.displaceable(w.priority, s, now) {
			cands = append(cands, s)
		}
	}
	// Best fit: the smallest single hold that covers the need; otherwise the holds idle
	// longest (least likely back) until it is covered. Ties go to the longest idle, then
	// to the session key, so the same state always displaces the same holds.
	sort.Slice(cands, func(i, j int) bool {
		a, b := cands[i], cands[j]
		if !a.lastDone.Equal(b.lastDone) {
			return a.lastDone.Before(b.lastDone)
		}
		return a.key < b.key
	})
	var single *cacheSession
	for _, s := range cands {
		if s.held >= need && (single == nil || s.held < single.held) {
			single = s
		}
	}
	if single != nil {
		return []displacement{{single, single.held}}, true
	}
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

// grantLocked admits waiters in order: priority, then tier, then longest wait. All
// queueing happens here; the engine is never sent more than it can run, and nothing
// until its pool is known. A waiter that
// does not fit reserves the room it still needs, so later waiters use only what is left
// beyond it and cannot starve it (unless it cannot run at all; see the reservation below).
func (q *AdmissionQueue) grantLocked() {
	if q.budget() == 0 || q.active >= q.cfg.MaxActive || len(q.waiters) == 0 {
		return
	}
	now := q.now()
	type scored struct {
		w      *admissionWaiter
		cached int
		tier   int
	}
	order := make([]scored, 0, len(q.waiters))
	for _, w := range q.waiters {
		q.keysLocked(w)
		cached := q.cachedLocked(w)
		order = append(order, scored{w, cached, q.tier(w, cached, now)})
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
	g := q.grantStateLocked(now)
	for _, o := range order {
		if q.active >= q.cfg.MaxActive {
			return
		}
		w := o.w
		plan, ok := q.fitLocked(w, now, reserved, g)
		if !ok {
			// A blocked waiter reserves the room it still needs, so later work cannot starve
			// it, unless it cannot run even once everything it may displace is gone: its
			// charge exceeds the budget less the holds of higher priority. Holding room back
			// for it would protect nothing.
			ever := q.budget()
			for p, h := range g.heldAbove {
				if p < w.priority {
					ever -= h
				}
			}
			if own := q.sessions[w.session]; own != nil && own.priority < w.priority {
				ever += own.held
			}
			if q.chargeLocked(w) <= ever {
				reserved += max(q.addsLocked(w)-g.free, 0)
			}
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
			w.report.ChargeTokens = q.liveCharge(w)
			w.report.PredictedCached = o.cached
			w.report.PrefillCostS = q.ownCost(w, o.cached)
			w.report.DisplacedTokens = displaced
			w.report.DisplaceCostS = cost
		}
		q.removeWaiterLocked(w)
		q.admitLocked(w, now)
		close(w.ready)
		g = q.grantStateLocked(now)
	}
}

func (q *AdmissionQueue) dropHoldLocked(s *cacheSession) {
	s.held = 0
	s.heldBlocks = kvmirror.PromptKeys{}
}

// syncHoldsLocked drops holds the engine no longer has, judged by the mirror while it is
// healthy (subscribed, and its replay complete). Run calls it once a second.
func (q *AdmissionQueue) syncHoldsLocked(now time.Time) {
	if !q.mirror.Healthy() {
		return
	}
	for _, s := range q.sessions {
		if s.held > 0 {
			q.judgeHoldLocked(s, now)
		}
	}
}

// judgeHoldLocked checks one hold against a healthy mirror. The first look records how
// much of it is cached, and a hold that later falls below half of that is gone (evicted,
// or lost to an engine restart or a resync whose replay did not reach it). Seeing nothing
// is evidence too when the mirror was reset since the turn was admitted: the engine's
// cache, or the mirror's view of it, started over, so the hold is gone. With no reset in
// between, a hold the mirror sees nothing of cannot be verified (no token IDs, blocks with
// extra hash keys, or a block or two, which the engine never serves whole). It is kept
// while firm and released after, and it is judged again on every pass, so a later reset
// releases it and blocks that show up later make it verified.
func (q *AdmissionQueue) judgeHoldLocked(s *cacheSession, now time.Time) {
	cached := q.mirror.Cached(s.heldBlocks)
	switch {
	case s.heldSeen > 0 && cached >= s.heldSeen/2:
		return
	case s.heldSeen <= 0 && cached > 0:
		s.heldSeen = cached
		return
	case s.heldSeen <= 0 && q.mirror.Generation() == s.heldGen && (s.heldSeen < 0 || q.firm(s, now)):
		s.heldSeen = 0
		return
	}
	q.dropHoldLocked(s)
	q.evictedHolds++
}

func (q *AdmissionQueue) admitLocked(w *admissionWaiter, now time.Time) {
	q.active++
	w.granted = true
	q.running[w] = struct{}{}
	w.tokens = nil
	w.blocks = w.blocks.Tail() // from now on only the prompt's own end can be a unit hit
	w.mirGen = q.mirror.Generation()
	if s := q.sessions[w.session]; s != nil {
		s.running++ // a session with a running request holds nothing (holding)
	}
}

// keysLocked rebuilds a waiter's mirror keys once the block size they were built at is
// no longer the mirror's (keys built before the engine's pool was read have none).
func (q *AdmissionQueue) keysLocked(w *admissionWaiter) {
	if w.tokens != nil && w.blocks.BlockSize != q.mirror.BlockSize() {
		w.blocks = q.mirror.NewPromptKeys(w.tokens)
	}
}

// finishLocked returns a request's charge and records what the engine reported. While
// a session has overlapping requests, the largest observation wins: a side call does not
// shrink what the main thread left cached, nor lend the hold its priority. Without usage
// (client gone mid-stream, upstream error), a request that streamed output had its prompt
// prefilled, and the size the proxy measured stands in. A busy period with nothing to
// show keeps the session's previous hold, whose prefix a retry resends; a session
// without one holds nothing.
func (q *AdmissionQueue) finishLocked(w *admissionWaiter, obs func() (CacheObservation, bool)) {
	now := q.now()
	delete(q.running, w)
	s := q.sessions[w.session]
	if s == nil {
		return
	}
	o, ok := obs()
	if ok && o.PromptTokens > 0 {
		s.outputs = pushInt(s.outputs, o.CompletionTokens)
	} else if g := w.generated(); g > 0 && w.prompt > 0 {
		o, ok = CacheObservation{PromptTokens: w.prompt, CompletionTokens: g}, true
	} else {
		ok = false
	}
	if fp := o.PromptTokens + o.CompletionTokens; ok && fp > s.busyFootprint {
		s.busyFootprint, s.busyPrompt, s.busyBlocks, s.busyPriority = fp, o.PromptTokens, w.blocks, w.priority
		s.busyGen = w.mirGen
	}
	if s.running > 0 {
		s.running--
	}
	if s.running > 0 {
		return
	}
	s.lastDone = now
	if s.busyFootprint > 0 {
		// The next turn resends the prompt, not the completion (thinking is dropped), so
		// the prompt is what stays useful in the cache.
		s.held = q.roundUp(s.busyPrompt)
		s.heldBlocks = s.busyBlocks
		s.heldSeen, s.heldGen = -1, s.busyGen
		s.priority = s.busyPriority
		if q.mirror.Healthy() {
			q.judgeHoldLocked(s, now) // the first look, while the blocks are fresh
		}
	}
	s.busyFootprint, s.busyPrompt, s.busyBlocks = 0, 0, kvmirror.PromptKeys{}
}

// CacheObservation is the engine's usage for one finished request.
type CacheObservation struct {
	PromptTokens     int
	CompletionTokens int
}

// PoolKnown reports whether the engine's KV pool is known; nothing is admitted until it is.
func (q *AdmissionQueue) PoolKnown() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.budget() > 0
}

// sharesHold: w's prompt is a prefix of its session's held prompt, by mirror keys.
func (q *AdmissionQueue) sharesHold(w *admissionWaiter) bool {
	s := q.sessions[w.session]
	if s == nil || s.held == 0 {
		return false
	}
	a, b := w.blocks, s.heldBlocks
	if len(a.Blocks) == 0 || a.BlockSize != b.BlockSize || len(a.Blocks) > len(b.Blocks) {
		return false
	}
	for i, k := range a.Blocks {
		if b.Blocks[i] != k {
			return false
		}
	}
	return true
}
