package proxy

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

// AdmissionConfig controls the request-lifetime gate in front of the engine: admitted
// work is bounded by the engine's KV pool (cache.go) and by MaxActive. Until the pool is
// known (the engine's metrics not read yet), requests wait.
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

type admissionWaiter struct {
	priority int
	enqueued time.Time // its place in the queue: arrival, or an earlier attempt's (retryMemory)
	arrived  time.Time
	sequence uint64
	ready    chan struct{}
	granted  bool

	session string
	prompt  int                 // prompt tokens (exact from /tokenize, else estimated)
	blocks  kvmirror.PromptKeys // mirror keys of the prompt
	tokens  []int64             // the prompt, kept only until its keys match the block size
	mirGen  uint64              // the mirror's generation when admitted
	maxOut  int                 // the request's output cap (0: none)
	seqs    int                 // sequences the engine runs for it (choices x prompts), at least 1
	expOut  int                 // expected output tokens, fixed at admission
	buffer  bool                // output unseen until the end: charged the cap
	gen     *OutputCounter
	report  *AdmissionReport
}

type AdmissionSnapshot struct {
	Active int
	Queued int
	// CacheUsed is the modelled cache occupancy in tokens.
	CacheUsed int
}

type AdmissionQueue struct {
	cfg       AdmissionConfig
	mu        sync.Mutex
	pool      int // engine block pool in tokens (SetPool); 0 until known
	blockSize int // engine block size (SetPool); 0 until known
	active    int
	sequence  uint64
	waiters   []*admissionWaiter
	// engineMetricsErrors counts failed reads of the engine's /metrics.
	engineMetricsErrors atomic.Uint64
	now                 func() time.Time

	sessions     map[string]*cacheSession
	running      map[*admissionWaiter]struct{}
	displaced    int
	evictedHolds int

	// Overcommit guard (guard.go).
	margin       int // blocks added to every running and waiting request's overhead
	marginRaises int
	lastRaise    time.Time
	lastSignal   time.Time
	lastPreempt  map[string]float64 // engine preemption counter per series (EngineSignals)
	preemptBlind bool               // the last read had no preemption counter

	// mirror says what the engine has cached (fed by its KV events, run.go).
	mirror *kvmirror.Mirror

	// Background cadences (Run, runEnginePool); fields so tests can shorten them.
	tickEvery, pruneEvery, poolEvery, poolRetryEvery time.Duration
}

func NewAdmissionQueue(cfg AdmissionConfig) *AdmissionQueue {
	cfg.applyDefaults()
	q := &AdmissionQueue{cfg: cfg, now: time.Now, sessions: map[string]*cacheSession{},
		running: map[*admissionWaiter]struct{}{}, mirror: kvmirror.New(),
		tickEvery: time.Second, pruneEvery: 10 * time.Minute, poolEvery: time.Minute, poolRetryEvery: 10 * time.Second}
	q.mirror.SetSpeculative(cfg.Cache.SpeculativeDecoding)
	return q
}

// Run re-evaluates the queue once a second: holds the engine dropped are released and
// expendable holds become displaceable without a request finishing first.
func (q *AdmissionQueue) Run(ctx context.Context) {
	t := time.NewTicker(q.tickEvery)
	defer t.Stop()
	prune := time.NewTicker(q.pruneEvery)
	defer prune.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			q.tick()
		case <-prune.C:
			q.mu.Lock()
			q.pruneSessionsLocked(q.now(), false)
			q.mu.Unlock()
		}
	}
}

type AdmissionRequest struct {
	Priority int

	// Session is empty for requests with no conversation identity; they are charged
	// while running and hold nothing afterwards.
	Session      string
	PromptTokens int
	// Tokens are the prompt's token IDs (from /tokenize), for the KV mirror; nil when the
	// prompt could not be tokenized.
	Tokens    []int64
	MaxTokens int
	// Sequences is how many sequences the engine runs for the request: OpenAI "n" times
	// the prompts in a batched /v1/completions prompt. Each has its own output.
	Sequences int
	// Generated, when set, counts output tokens so far; the request's charge grows with
	// it once it passes the expected output.
	Generated *OutputCounter
	// Buffered marks a request whose output is not seen until it finishes (not
	// streaming): it is charged its whole output cap from admission (MaxTokens, else
	// default_output_tokens), since nothing raises its charge while it runs.
	Buffered bool
	// Observe is called at release and reports the engine's usage for the request.
	Observe func() (CacheObservation, bool)
	// Report, when set, is filled at admission time for the request log.
	Report *AdmissionReport
}

// AdmissionReport is filled for the request log at admission (read only after Acquire
// returns).
type AdmissionReport struct {
	ChargeTokens    int
	PredictedCached int
	PrefillCostS    float64
	DisplacedTokens int
	DisplaceCostS   float64
}

// Admitted describes an admission for the request log.
type Admitted struct {
	Wait      time.Duration     // time queued (0 when admitted at once)
	OnArrival AdmissionSnapshot // the queue as the request found it
}

// Acquire waits for an execution permit until it is granted or ctx ends (ok false).
// The returned release must be called exactly once.
// testHookGaveUp runs when a waiter sees its context end, before it takes the lock;
// tests use it to grant the waiter in between. Nil outside tests.
var testHookGaveUp func()

func (q *AdmissionQueue) Acquire(ctx context.Context, ar AdmissionRequest) (release func(), adm Admitted, ok bool) {
	observe := ar.Observe
	if observe == nil {
		observe = func() (CacheObservation, bool) { return CacheObservation{}, false }
	}
	w, granted, snap := q.enqueue(ar)
	if granted {
		return q.releaseFunc(w, observe), Admitted{OnArrival: snap}, true
	}
	select {
	case <-w.ready:
		return q.releaseFunc(w, observe), Admitted{q.now().Sub(w.arrived), snap}, true
	case <-ctx.Done():
		if testHookGaveUp != nil {
			testHookGaveUp()
		}
		q.mu.Lock()
		if w.granted {
			q.mu.Unlock()
			q.release(w, observe)
		} else {
			if s := q.sessions[w.session]; s != nil {
				s.gaveUpSince, s.gaveUpAt = w.arrived, q.now()
			}
			q.removeWaiterLocked(w)
			q.grantLocked()
			q.mu.Unlock()
		}
		return nil, Admitted{q.now().Sub(w.arrived), snap}, false
	}
}

// enqueue registers a waiter and admits it at once if it fits.
func (q *AdmissionQueue) enqueue(ar AdmissionRequest) (*admissionWaiter, bool, AdmissionSnapshot) {
	// Keys are built outside the queue's lock; while the block size is unknown they are
	// built again when it is (keysLocked).
	keys := q.mirror.NewPromptKeys(ar.Tokens)
	tokens := ar.Tokens
	if keys.BlockSize > 0 {
		tokens = nil
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	now := q.now()
	w := &admissionWaiter{
		priority: ar.Priority, enqueued: now, arrived: now,
		sequence: q.sequence, ready: make(chan struct{}), session: ar.Session,
		prompt: max(ar.PromptTokens, 0), blocks: keys, tokens: tokens, maxOut: ar.MaxTokens, seqs: max(ar.Sequences, 1),
		report: ar.Report, gen: ar.Generated, buffer: ar.Buffered,
	}
	q.sequence++
	// A client that timed out and retried keeps its place: its wait counts from the
	// previous attempt's arrival (not from what that attempt inherited, so giving up and
	// returning again and again cannot push a session ahead of everyone), so a retry is
	// not sent to the back.
	if s := q.session(ar.Session, now); s != nil {
		if !s.gaveUpAt.IsZero() && now.Sub(s.gaveUpAt) < retryMemory {
			w.enqueued = s.gaveUpSince
		}
		s.gaveUpAt = time.Time{}
		if s.waiting == 0 {
			s.waitingSince = now
		}
		s.waiting++
	}
	snap := q.snapshotLocked()
	q.waiters = append(q.waiters, w)
	q.grantLocked()
	return w, w.granted, snap
}

func (q *AdmissionQueue) releaseFunc(w *admissionWaiter, observe func() (CacheObservation, bool)) func() {
	var once sync.Once
	return func() { once.Do(func() { q.release(w, observe) }) }
}

func (q *AdmissionQueue) release(w *admissionWaiter, observe func() (CacheObservation, bool)) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.active > 0 {
		q.active--
	}
	q.finishLocked(w, observe)
	q.grantLocked()
}

func (q *AdmissionQueue) snapshotLocked() AdmissionSnapshot {
	return AdmissionSnapshot{Active: q.active, Queued: len(q.waiters), CacheUsed: q.cacheUsedLocked(q.now())}
}

// removeWaiterLocked takes a waiter out of the queue, granted or given up; it is the
// only way out, so a session's waiting count always matches its queued requests.
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

// tick is the queue's once-a-second pass: the guard decays, holds are judged against the
// mirror (syncHoldsLocked; once a second, not on every grant, as it looks at every hold),
// and waiters are re-evaluated.
func (q *AdmissionQueue) tick() {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := q.now()
	q.guardDecayLocked(now)
	q.syncHoldsLocked(now)
	q.grantLocked()
}
