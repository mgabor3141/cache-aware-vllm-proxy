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
)

// requestMetrics counts finished API requests.
type requestMetrics struct {
	mu           sync.Mutex
	requests     map[routeStatus]uint64
	promptTokens uint64
	cachedTokens uint64
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
	m.promptTokens += uint64(max(ev.PromptTokens, 0))
	m.cachedTokens += uint64(max(ev.CachedPromptTokens, 0))
}

// admissionMetrics is an aggregate view of the admission queue.
type admissionMetrics struct {
	queued, active                 int
	usedTokens, budgetTokens, held int
	marginBlocks, marginRaises     int
	mirrorConnected                bool
}

func (q *AdmissionQueue) metrics() admissionMetrics {
	q.mu.Lock()
	now := q.now()
	m := admissionMetrics{
		queued: len(q.waiters), active: q.active,
		usedTokens: q.cacheUsedLocked(now), budgetTokens: q.budget(), held: q.heldLocked(now),
		marginBlocks: q.margin, marginRaises: q.marginRaises,
	}
	mirror := q.mirror
	q.mu.Unlock()
	if mirror != nil {
		m.mirrorConnected = mirror.Healthy()
	}
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
	gauge("cavp_queue_length", "Requests waiting for admission.", a.queued)
	gauge("cavp_active_requests", "Admitted requests running on the engine.", a.active)
	gauge("cavp_cache_used_tokens", "Modelled KV cache use: running requests and held contexts.", a.usedTokens)
	gauge("cavp_cache_budget_tokens", "KV cache budget (the engine's pool; 0 until known).", a.budgetTokens)
	gauge("cavp_cache_held_tokens", "Tokens held for idle conversations.", a.held)
	gauge("cavp_guard_margin_blocks", "Overcommit guard margin added to each running request.", a.marginBlocks)
	counter("cavp_guard_raises_total", "Times the overcommit guard raised its margin.", uint64(a.marginRaises))
	gauge("cavp_kv_mirror_connected", "1 while the KV event mirror is subscribed.", b2i(a.mirrorConnected))

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
	prompt, cached := p.metrics.promptTokens, p.metrics.cachedTokens
	p.metrics.mu.Unlock()
	counter("cavp_prompt_tokens_total", "Prompt tokens of finished requests, as the engine reported them.", prompt)
	counter("cavp_cached_prompt_tokens_total", "Of those, tokens served from the engine's prefix cache.", cached)
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
