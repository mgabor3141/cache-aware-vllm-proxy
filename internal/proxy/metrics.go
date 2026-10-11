// GET /metrics: aggregate state in the Prometheus text format. Keyless like /readyz,
// so it carries totals only, never per-session or per-caller data.
package proxy

import (
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"
)

// requestMetrics counts finished API requests.
type requestMetrics struct {
	mu             sync.Mutex
	requests       map[routeStatus]uint64
	promptTokens   uint64
	cachedTokens   uint64
	tokenizeErrors uint64
	queueWait      histogram
}

// queueWaitBuckets are the upper bounds, in seconds, of the queue wait histogram.
var queueWaitBuckets = []float64{0.1, 0.5, 1, 5, 15, 60, 300, 900, 3600}

// histogram is a Prometheus histogram over queueWaitBuckets.
type histogram struct {
	counts []uint64 // per bucket, not cumulative; one more for +Inf
	sum    float64
	n      uint64
}

func (h *histogram) observe(v float64) {
	if h.counts == nil {
		h.counts = make([]uint64, len(queueWaitBuckets)+1)
	}
	i := sort.SearchFloat64s(queueWaitBuckets, v)
	h.counts[i]++
	h.sum += v
	h.n++
}

func (h *histogram) write(w io.Writer, name, help string) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name)
	var c uint64
	for i, b := range queueWaitBuckets {
		if h.counts != nil {
			c += h.counts[i]
		}
		fmt.Fprintf(w, "%s_bucket{le=%q} %d\n", name, strconv.FormatFloat(b, 'g', -1, 64), c)
	}
	fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n%s_sum %g\n%s_count %d\n", name, h.n, name, h.sum, name, h.n)
}

// admitted records how long an admitted request waited for its permit.
func (m *requestMetrics) admitted(wait time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queueWait.observe(wait.Seconds())
}

// tokenizeFailed counts a prompt that /tokenize could not size (priced from its bytes).
func (m *requestMetrics) tokenizeFailed() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenizeErrors++
}

type routeStatus struct {
	route  string
	status int
}

func (m *requestMetrics) observe(route string, ev *Event) {
	if route == "" {
		route = "none"
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.requests == nil {
		m.requests = map[routeStatus]uint64{}
	}
	m.requests[routeStatus{route, ev.Status}]++
	// The hit ratio counts only requests whose engine reported cache details (vLLM
	// --enable-prompt-tokens-details): without them a hit is unknown, not a miss.
	if ev.UsageDetails {
		m.promptTokens += uint64(max(ev.PromptTokens, 0))
		m.cachedTokens += uint64(max(ev.CachedPromptTokens, 0))
	}
}

// admissionMetrics is an aggregate view of the admission queue.
type admissionMetrics struct {
	queued, active                 int
	usedTokens, budgetTokens, held int
	marginBlocks, marginRaises     int
	displaced, evictedHolds        int
	mirrorConnected                bool
	oldestWait                     time.Duration
	engineMetricsErrors            uint64
}

func (q *AdmissionQueue) metrics() admissionMetrics {
	q.mu.Lock()
	now := q.now()
	m := admissionMetrics{
		queued: len(q.waiters), active: q.active,
		usedTokens: q.cacheUsedLocked(now), budgetTokens: q.budget(), held: q.heldLocked(now),
		marginBlocks: q.margin, marginRaises: q.marginRaises,
		displaced: q.displaced, evictedHolds: q.evictedHolds,
	}
	for _, w := range q.waiters {
		m.oldestWait = max(m.oldestWait, now.Sub(w.arrived))
	}
	m.engineMetricsErrors = q.engineMetricsErrors.Load()
	mirror := q.mirror
	q.mu.Unlock()
	m.mirrorConnected = mirror.Healthy()
	return m
}

func (p *Proxy) serveMetrics(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	a := p.admission.metrics()
	gauge := func(name, help string, v int) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %d\n", name, help, name, name, v)
	}
	counter := func(name, help string, v uint64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, v)
	}
	gauge("cavp_engine_ready", "1 while the engine's readiness probe succeeds.", b2i(p.ready.Ready()))
	gauge("cavp_ready_wait_requests", "Requests waiting for the engine to become ready (not in cavp_queue_length).", int(p.ready.waiting.Load()))
	gauge("cavp_queue_length", "Requests waiting for admission.", a.queued)
	fmt.Fprintf(w, "# HELP cavp_queue_oldest_wait_seconds How long the longest waiting request has waited for admission (0: none waiting).\n# TYPE cavp_queue_oldest_wait_seconds gauge\ncavp_queue_oldest_wait_seconds %g\n", a.oldestWait.Seconds())
	gauge("cavp_active_requests", "Admitted requests running on the engine.", a.active)
	gauge("cavp_cache_used_tokens", "Modelled KV cache use: running requests and held contexts.", a.usedTokens)
	gauge("cavp_cache_budget_tokens", "KV cache budget (the engine's pool; 0 until known).", a.budgetTokens)
	gauge("cavp_cache_held_tokens", "Tokens held for conversations, idle or with a request running.", a.held)
	gauge("cavp_guard_margin_blocks", "Overcommit guard margin added to each running request.", a.marginBlocks)
	counter("cavp_guard_raises_total", "Times the overcommit guard raised its margin.", uint64(a.marginRaises))
	counter("cavp_cache_displaced_tokens_total", "Held tokens released to make room for other work.", uint64(a.displaced))
	counter("cavp_cache_holds_evicted_total", "Holds released because the KV mirror no longer saw them cached.", uint64(a.evictedHolds))
	gauge("cavp_kv_mirror_connected", "1 while the KV event mirror is subscribed and caught up.", b2i(a.mirrorConnected))
	counter("cavp_engine_metrics_errors_total", "Failed reads of the engine's /metrics (KV pool and preemptions); the guard is blind while they fail.", a.engineMetricsErrors)

	p.metrics.mu.Lock()
	keys := make([]routeStatus, 0, len(p.metrics.requests))
	for k := range p.metrics.requests {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].route != keys[j].route {
			return keys[i].route < keys[j].route
		}
		return keys[i].status < keys[j].status
	})
	io.WriteString(w, "# HELP cavp_requests_total API requests by route and logged status.\n# TYPE cavp_requests_total counter\n")
	for _, k := range keys {
		fmt.Fprintf(w, "cavp_requests_total{route=%q,status=%q} %d\n", k.route, strconv.Itoa(k.status), p.metrics.requests[k])
	}
	p.metrics.queueWait.write(w, "cavp_queue_wait_seconds", "Time admitted requests waited for admission.")
	tokErr := p.metrics.tokenizeErrors
	prompt, cached := p.metrics.promptTokens, p.metrics.cachedTokens
	p.metrics.mu.Unlock()
	counter("cavp_prompt_tokens_total", "Prompt tokens of finished requests whose engine reported cache details.", prompt)
	counter("cavp_cached_prompt_tokens_total", "Of those, tokens served from the engine's prefix cache.", cached)
	counter("cavp_tokenize_errors_total", "Prompts /tokenize could not size, priced from their bytes instead.", tokErr)
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
