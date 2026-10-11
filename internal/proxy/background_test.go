package proxy

import (
	"context"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

const testPoolMetrics = `vllm:cache_config_info{block_size="3200",num_gpu_blocks="201",prefix_match_unit="64"} 1.0`

// startBackground starts every loop the proxy needs: the engine pool becomes known, the
// probe finds the engine ready, and the admission pass forgets idle sessions.
func TestStartBackgroundWiring(t *testing.T) {
	eng := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/metrics":
			io.WriteString(w, testPoolMetrics)
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"m"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	cfg := &Config{
		Upstream:  eng.URL,
		Auth:      auth.Config{Mode: "static", Keys: []auth.StaticKey{{Key: testKey, User: "alice"}}},
		Admission: AdmissionConfig{MaxActive: 2, Cache: CacheConfig{KVEventsEndpoint: "tcp://127.0.0.1:1"}},
		Readiness: ReadinessConfig{Interval: 20 * time.Millisecond},
	}
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	prx, err := NewProxy(cfg, NewEventLogger(&logBuf{}))
	if err != nil {
		t.Fatal(err)
	}
	q := prx.admission
	q.tickEvery, q.pruneEvery = 10*time.Millisecond, 20*time.Millisecond
	q.mu.Lock()
	q.sessions["old"] = &cacheSession{key: "old", lastSeen: q.now().Add(-25 * time.Hour)}
	q.mu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startBackground(ctx, prx, cfg)
	waitFor(t, "the engine pool", q.PoolKnown)
	waitFor(t, "the engine ready", prx.ready.Ready)
	waitFor(t, "the idle session forgotten", func() bool {
		q.mu.Lock()
		defer q.mu.Unlock()
		_, ok := q.sessions["old"]
		return !ok
	})
}

// The admission pass acts with no request finishing: a hold the engine dropped is
// released and the waiter it blocked gets in.
func TestAdmissionRunReleasesDroppedHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	q.tickEvery = 10 * time.Millisecond
	m := q.mirror
	m.SetConnected(true)
	prompt := testPrompt(20, 0)
	storePrompt(m, 1, prompt, 1)
	tr := q.startReq("a", 0, len(prompt), prompt)
	tr.finish(mustAdmit(t, tr, "a's turn"), len(prompt), 100)
	q.tick()
	if q.sessionHeld("a") == 0 {
		t.Fatal("setup: a holds nothing")
	}
	big := q.start("b", 5, 640000) // the whole pool: fits only once the hold is gone
	if rel := big.waitAdmitted(settle); rel != nil {
		rel()
		t.Fatal("setup: the waiter fits beside the hold")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go q.Run(ctx)
	g := 5
	var rm []any
	for i := 0; i < 20; i++ {
		rm = append(rm, uint64(i+1))
	}
	m.Apply(2, []kvmirror.Event{{Type: "BlockRemoved", BlockHashes: rm, GroupIdx: &g}})
	rel := big.waitAdmitted(2 * time.Second)
	if rel == nil {
		t.Fatalf("Run never re-evaluated: held=%d queued=%d", q.sessionHeld("a"), q.Snapshot().Queued)
	}
	rel()
}

func TestFetchEnginePoolShapes(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		wantErr bool
	}{
		{"labels", 200, testPoolMetrics, false},
		{"no cache_config_info", 200, "vllm:x 1\n", true},
		{"HTTP 500", 500, "nope", true},
		{"zero blocks", 200, `vllm:cache_config_info{block_size="3200",num_gpu_blocks="0"} 1.0`, true},
	}
	for _, c := range cases {
		srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(c.status)
			io.WriteString(w, c.body)
		}))
		b, s, u, err := fetchEnginePool(context.Background(), srv.URL)
		if c.wantErr != (err != nil) {
			t.Errorf("%s: %d x %d unit %d err %v", c.name, b, s, u, err)
		}
		if !c.wantErr && (b != 201 || s != 3200 || u != 64) {
			t.Errorf("%s: %d x %d unit %d", c.name, b, s, u)
		}
	}
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	addr := l.Addr().String()
	l.Close()
	if _, _, _, err := fetchEnginePool(context.Background(), "http://"+addr); err == nil {
		t.Error("a refused connection read as a pool")
	}
}

// The pool loop retries until the pool is known, keeps it through failed reads, and
// follows a changed pool.
func TestEnginePoolLoopFollowsChangesAndKeepsLast(t *testing.T) {
	var body atomic.Value
	body.Store("vllm:x 1\n")
	var failing atomic.Bool
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if failing.Load() {
			w.WriteHeader(500)
			return
		}
		io.WriteString(w, body.Load().(string))
	}))
	q := NewAdmissionQueue(AdmissionConfig{MaxActive: 1})
	q.poolEvery, q.poolRetryEvery = 20*time.Millisecond, 10*time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go runEnginePool(ctx, srv.URL, q)
	time.Sleep(50 * time.Millisecond)
	if q.PoolKnown() {
		t.Fatal("known without cache_config_info")
	}
	body.Store(testPoolMetrics)
	waitFor(t, "the pool becoming known", q.PoolKnown)
	failing.Store(true)
	time.Sleep(100 * time.Millisecond)
	if !q.PoolKnown() {
		t.Fatal("a failed read cleared a known pool")
	}
	if q.engineMetricsErrors.Load() == 0 {
		t.Error("failed reads not counted")
	}
	body.Store(`vllm:cache_config_info{block_size="1600",num_gpu_blocks="401"} 1.0`)
	failing.Store(false)
	waitFor(t, "the changed pool", func() bool {
		q.mu.Lock()
		defer q.mu.Unlock()
		return q.pool == 400*1600 && q.blockSize == 1600
	})
}

// The readiness loop counts refused connections as gone, and a busy engine (an error
// answer or a timeout) never.
func TestReadinessRunCountsGoneButNeverBusy(t *testing.T) {
	const every = 30 * time.Millisecond
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	gone := newReadiness("http://"+addr, every)
	go gone.Run(ctx)
	waitFor(t, "gone time", func() bool { return gone.goneFor() >= 3*every })

	for name, busy := range map[string]http.HandlerFunc{
		"503":  func(w http.ResponseWriter, r *http.Request) { http.Error(w, "loading", 503) },
		"hang": func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() },
	} {
		rd := newReadiness(newTestServer(t, busy).URL, every)
		rd.client.Timeout = 2 * every
		go rd.Run(ctx)
		time.Sleep(400 * time.Millisecond)
		if rd.Ready() || rd.goneFor() != 0 {
			t.Errorf("%s: ready=%v goneFor=%v", name, rd.Ready(), rd.goneFor())
		}
	}
}
