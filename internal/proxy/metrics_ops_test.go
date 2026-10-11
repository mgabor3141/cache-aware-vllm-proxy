package proxy

import (
	"context"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

func metricValueOf(t *testing.T, h *harness, name string) float64 {
	t.Helper()
	_, m := h.do("GET", "/metrics", "", nil)
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` (\S+)$`)
	g := re.FindStringSubmatch(m)
	if g == nil {
		t.Fatalf("no %s in\n%s", name, m)
	}
	v, _ := strconv.ParseFloat(g[1], 64)
	return v
}

// Requests waiting for the engine show in a gauge of their own (operator #5, #22).
func TestMetricsReadyWaitGauge(t *testing.T) {
	h := newHarness(t, nil, nil)
	h.p.ready.set(false, "test")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		if resp, err := h.request(ctx, "POST", "/v1/chat/completions", chatBody, nil); err == nil {
			resp.Body.Close()
		}
	}()
	waitFor(t, "ready wait gauge", func() bool { return metricValueOf(t, h, "cavp_ready_wait_requests") == 1 })
	cancel()
	<-done
	waitFor(t, "gauge back to 0", func() bool { return metricValueOf(t, h, "cavp_ready_wait_requests") == 0 })
}

// The queue's oldest wait is a gauge and admitted waits a histogram (operator #26).
func TestMetricsQueueWaitAgeAndHistogram(t *testing.T) {
	h := newHarness(t, nil, func(c *Config) { c.Admission.MaxActive = 1 })
	rel, _, _ := h.p.admission.Acquire(context.Background(), AdmissionRequest{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		h.do("POST", "/v1/chat/completions", chatBody, nil)
	}()
	waitFor(t, "a waiter", func() bool { return h.p.admission.Snapshot().Queued == 1 })
	time.Sleep(150 * time.Millisecond)
	if v := metricValueOf(t, h, "cavp_queue_oldest_wait_seconds"); v < 0.1 {
		t.Fatalf("oldest wait %g while a request waited 150 ms", v)
	}
	rel()
	<-done
	if v := metricValueOf(t, h, "cavp_queue_oldest_wait_seconds"); v != 0 {
		t.Fatalf("oldest wait %g with nothing waiting", v)
	}
	// The harness's own Acquire is not a request; the one request is counted.
	if v := metricValueOf(t, h, "cavp_queue_wait_seconds_count"); v != 1 {
		t.Fatalf("queue wait count %g", v)
	}
	_, m := h.do("GET", "/metrics", "", nil)
	if !strings.Contains(m, `cavp_queue_wait_seconds_bucket{le="0.1"} 0`) || !strings.Contains(m, `cavp_queue_wait_seconds_bucket{le="0.5"} 1`) {
		t.Fatalf("buckets:\n%s", m)
	}
}

// A failed /tokenize and a failed read of the engine's /metrics are counted
// (operator #24, #4).
func TestMetricsCountsTokenizeAndEngineMetricsFailures(t *testing.T) {
	h := newHarnessEngine(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tokenize", "/metrics":
			http.Error(w, "down", http.StatusInternalServerError)
		default:
			w.Write([]byte(realVLLMChatResponse))
		}
	}), nil)
	h.do("POST", "/v1/chat/completions", chatBody, nil)
	if v := metricValueOf(t, h, "cavp_tokenize_errors_total"); v != 1 {
		t.Fatalf("tokenize errors %g", v)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go runEnginePreemptions(ctx, h.eng.URL, h.p.admission)
	time.Sleep(1300 * time.Millisecond)
	if v := metricValueOf(t, h, "cavp_engine_metrics_errors_total"); v < 1 {
		t.Fatalf("engine metrics errors %g after a failed read", v)
	}
}

// The KV mirror's counters reach /metrics, so a mirror that is connected but skipping
// every event (offloaded tiers, unreadable hashes) is visible (load3 #100 F9, F10).
func TestMetricsExportMirrorStats(t *testing.T) {
	h := newHarness(t, nil, nil)
	g, cpu := 0, "CPU"
	toks := make([]int64, 16)
	h.p.admission.mirror.Apply(-1, []kvmirror.Event{
		{Type: "BlockStored", BlockHashes: []any{uint64(1)}, TokenIDs: toks, BlockSize: 16, GroupIdx: &g, Kind: "full_attention"},
		{Type: "BlockStored", BlockHashes: []any{uint64(2)}, TokenIDs: toks, BlockSize: 16, GroupIdx: &g, Medium: &cpu},
	})
	for name, want := range map[string]float64{
		"cavp_kv_mirror_blocks":                              1,
		"cavp_kv_mirror_unsupported":                         0,
		`cavp_kv_mirror_events_total{kind="stored"}`:         1,
		`cavp_kv_mirror_events_total{kind="skipped_medium"}`: 1,
		`cavp_kv_mirror_events_total{kind="skipped_hash"}`:   0,
	} {
		if got := metricValueOf(t, h, name); got != want {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
}
