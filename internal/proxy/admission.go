package proxy

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

// AdmissionConfig controls the request-lifetime gate in front of the LLM upstream.
// With cache admission (cache.go) the engine KV pool bounds admitted work; until the
// pool is known, or with it off, only MaxActive does, and waiters run by priority
// (lowest value first, as in vLLM) and then arrival.
type AdmissionConfig struct {
	// A waiter waits until it is admitted or its client gives up; there is no queue
	// timeout. Agents retry, and a timeout firing for many waiters at once would only
	// add load.
	MaxActive int `yaml:"max_active"`

	// Cache bounds admitted work by the engine's prefix cache; see cache.go.
	Cache CacheConfig `yaml:"cache"`
}

func (c *AdmissionConfig) applyDefaults() {
	if c.MaxActive <= 0 {
		c.MaxActive = 1
	}
	c.Cache.applyDefaults()
}

func (c AdmissionConfig) validate() error { return c.Cache.validate() }

// SetMirror attaches the KV event mirror that says what the engine has cached.
func (q *AdmissionQueue) SetMirror(m *kvmirror.Mirror) {
	q.mu.Lock()
	q.mirror = m
	q.mu.Unlock()
}

type admissionWaiter struct {
	priority int
	enqueued time.Time
	sequence uint64
	ready    chan struct{}
	granted  bool

	session string
	prompt  int                 // prompt tokens (exact from /tokenize, else estimated)
	blocks  kvmirror.PromptKeys // mirror keys of the prompt
	charge  int                 // cache tokens at admission (prompt, expected output, state)
	expOut  int                 // expected output tokens
	gen     *atomic.Int64
	report  *AdmissionReport
}

type AdmissionSnapshot struct {
	Active int
	Queued int
	// CacheUsed is the modelled cache occupancy in tokens (0 when cache admission is off).
	CacheUsed int
}

type AdmissionQueue struct {
	cfg      AdmissionConfig
	mu       sync.Mutex
	pool     int // engine block pool in tokens (SetPool); 0 until known
	active   int
	sequence uint64
	waiters  []*admissionWaiter
	now      func() time.Time

	sessions     map[string]*cacheSession
	running      map[*admissionWaiter]struct{}
	engineUsed   int // tokens the engine's running requests occupy (SetEngineUsed)
	displaced    int
	evictedHolds int

	// Overcommit guard (guard.go).
	margin       int // blocks added to every running request's overhead
	marginRaises int
	lastRaise    time.Time
	lastSignal   time.Time
	lastPreempt  float64 // engine preemption counter; -1 until first read
	mirror       *kvmirror.Mirror
}

func NewAdmissionQueue(cfg AdmissionConfig) *AdmissionQueue {
	cfg.applyDefaults()
	return &AdmissionQueue{cfg: cfg, now: time.Now, sessions: map[string]*cacheSession{}, pool: cfg.Cache.Tokens,
		running: map[*admissionWaiter]struct{}{}, lastPreempt: -1}
}

// Run re-evaluates waiters once a second, so idle holds whose return grew unlikely are
// released without needing a request to finish first.
func (q *AdmissionQueue) Run(ctx context.Context) {
	if !q.cfg.Cache.enabled() {
		return
	}
	t := time.NewTicker(time.Second)
	defer t.Stop()
	prune := time.NewTicker(10 * time.Minute)
	defer prune.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			q.mu.Lock()
			q.guardDecayLocked(q.now())
			if len(q.waiters) > 0 {
				q.grantLocked()
			}
			q.mu.Unlock()
		case <-prune.C:
			q.mu.Lock()
			q.pruneSessionsLocked(q.now(), false)
			q.mu.Unlock()
		}
	}
}

type AdmissionRequest struct {
	Priority int

	// Cache admission inputs. Session is empty for requests with no conversation
	// identity; they are charged while running and hold nothing afterwards.
	Session      string
	PromptTokens int
	Blocks       kvmirror.PromptKeys
	MaxTokens    int
	// Generated, when set, counts output tokens so far (an upper bound); the request's
	// charge grows with it instead of reserving its whole expected output.
	Generated *atomic.Int64
	// Observe is called at release and reports the engine's usage for the request.
	Observe func() (CacheObservation, bool)
	// Report, when set, is filled at admission time for the request log.
	Report *AdmissionReport
}

// AdmissionReport is filled for the request log: the charge at arrival, the rest at
// admission (read only after Acquire returns).
type AdmissionReport struct {
	ChargeTokens    int
	PredictedCached int
	PrefillCostS    float64
	DisplacedTokens int
	DisplaceCostS   float64
}

// Acquire waits for an LLM execution permit. The returned release function
// must be called exactly once. queued is false for immediate admission.
func (q *AdmissionQueue) Acquire(ctx context.Context, ar AdmissionRequest) (release func(), queued bool, wait time.Duration, snap AdmissionSnapshot, ok bool) {
	observe := ar.Observe
	if observe == nil {
		observe = func() (CacheObservation, bool) { return CacheObservation{}, false }
	}
	q.mu.Lock()
	now := q.now()
	w := &admissionWaiter{
		priority: ar.Priority, enqueued: now,
		sequence: q.sequence, ready: make(chan struct{}), session: ar.Session,
		prompt: max(ar.PromptTokens, 0), blocks: ar.Blocks, report: ar.Report,
		gen: ar.Generated,
	}
	q.sequence++
	if q.cfg.Cache.enabled() {
		s := q.session(ar.Session, now)
		// A client that timed out and retried keeps its place: its wait counts from the
		// first attempt, so a retry is not sent to the back.
		if ar.Session != "" && !s.gaveUpAt.IsZero() && now.Sub(s.gaveUpAt) < retryMemory {
			w.enqueued = s.gaveUpSince
		}
		s.gaveUpAt = time.Time{}
		w.expOut = q.expectedOutput(s, ar.MaxTokens)
		w.charge = q.liveCharge(w)
		if ar.Report != nil {
			ar.Report.ChargeTokens = w.charge
		}
	}
	snap = q.snapshotLocked()

	q.waiters = append(q.waiters, w)
	if s := q.sessions[w.session]; s != nil {
		s.waiting++
	}
	q.grantLocked()
	if w.granted {
		q.mu.Unlock()
		return q.releaseFunc(w, observe), false, 0, snap, true
	}
	q.mu.Unlock()

	select {
	case <-w.ready:
		return q.releaseFunc(w, observe), true, q.now().Sub(w.enqueued), snap, true
	case <-ctx.Done():
		q.mu.Lock()
		if w.granted {
			q.mu.Unlock()
			q.release(w, observe)
		} else {
			if s := q.sessions[w.session]; s != nil && w.session != "" {
				s.gaveUpSince, s.gaveUpAt = w.enqueued, q.now()
			}
			q.removeWaiterLocked(w)
			q.grantLocked()
			q.mu.Unlock()
		}
		return nil, true, q.now().Sub(w.enqueued), snap, false
	}
}

func (q *AdmissionQueue) releaseFunc(w *admissionWaiter, observe func() (CacheObservation, bool)) func() {
	var once sync.Once
	return func() { once.Do(func() { q.release(w, observe) }) }
}

func (q *AdmissionQueue) release(w *admissionWaiter, observe func() (CacheObservation, bool)) {
	q.mu.Lock()
	if q.active > 0 {
		q.active--
	}
	q.finishLocked(w, observe)
	q.grantLocked()
	q.mu.Unlock()
}

func (q *AdmissionQueue) snapshotLocked() AdmissionSnapshot {
	snap := AdmissionSnapshot{Active: q.active, Queued: len(q.waiters)}
	if q.cfg.Cache.enabled() {
		snap.CacheUsed = q.cacheUsedLocked(q.now())
	}
	return snap
}

func (q *AdmissionQueue) grantLocked() {
	if q.cfg.Cache.enabled() && q.budget() > 0 {
		q.grantCacheLocked()
		return
	}
	for q.active < q.cfg.MaxActive && len(q.waiters) > 0 {
		now := q.now()
		best := 0
		for i := 1; i < len(q.waiters); i++ {
			c, b := q.waiters[i], q.waiters[best]
			if c.priority < b.priority || (c.priority == b.priority && c.sequence < b.sequence) {
				best = i
			}
		}
		w := q.waiters[best]
		q.waiters = append(q.waiters[:best], q.waiters[best+1:]...)
		q.admitLocked(w, now)
		close(w.ready)
	}
}

func (q *AdmissionQueue) removeWaiterLocked(target *admissionWaiter) {
	for i, w := range q.waiters {
		if w == target {
			if s := q.sessions[w.session]; s != nil && s.waiting > 0 {
				s.waiting--
			}
			q.waiters = append(q.waiters[:i], q.waiters[i+1:]...)
			return
		}
	}
}

func (q *AdmissionQueue) Snapshot() AdmissionSnapshot {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.snapshotLocked()
}
