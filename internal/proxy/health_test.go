package proxy

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func testHealthConfig() RetryConfig {
	return RetryConfig{
		RecentSuccessWindow:   100 * time.Millisecond,
		RecoveryTimeout:       60 * time.Millisecond,
		HealthyProbeInterval:  10 * time.Millisecond,
		RecoveryProbeInterval: 5 * time.Millisecond,
		DownProbeInterval:     10 * time.Millisecond,
	}
}

func TestUpstreamHealth_RecentlyHealthyGetsOneRecoveryEpisode(t *testing.T) {
	h := newUpstreamHealth("http://unused", testHealthConfig(), nil)
	h.recordSuccess("test")
	h.recordFailure("test_failure", false)
	if got := h.State(); got != upstreamRecovering {
		t.Fatalf("state=%s, want recovering", got)
	}

	time.Sleep(70 * time.Millisecond)
	h.recordFailure("deadline", false)
	if got := h.State(); got != upstreamDown {
		t.Fatalf("state=%s, want down", got)
	}

	// A later 503 cannot give new callers another long wait. Only a real
	// readiness success resets the outage episode.
	h.recordFailure("another_503", true)
	if got := h.State(); got != upstreamDown {
		t.Fatalf("state=%s, want down after exhausted episode", got)
	}
	h.recordSuccess("recovered")
	if got := h.State(); got != upstreamHealthy {
		t.Fatalf("state=%s, want healthy", got)
	}
}

func TestUpstreamHealth_ColdRefusalFailsFastButCold503GetsOneChance(t *testing.T) {
	h := newUpstreamHealth("http://unused", testHealthConfig(), nil)
	h.recordFailure("connection_refused", false)
	if got := h.State(); got != upstreamDown {
		t.Fatalf("state=%s, want down", got)
	}
	if err := h.WaitReady(context.Background()); !errors.Is(err, ErrUpstreamUnavailable) {
		t.Fatalf("WaitReady error=%v, want ErrUpstreamUnavailable", err)
	}

	h.recordFailure("router_loading", true)
	if got := h.State(); got != upstreamRecovering {
		t.Fatalf("state=%s, want recovering after first cold 503", got)
	}
}

func TestUpstreamHealth_ReadinessDoesNotResetActiveOutageClock(t *testing.T) {
	h := newUpstreamHealth("http://unused", testHealthConfig(), nil)
	h.recordSuccess("request")
	h.recordFailure("request_503", true)

	h.mu.Lock()
	deadline := h.recoveryDeadline
	h.mu.Unlock()
	h.recordSuccess("readiness")
	h.recordFailure("request_still_503", true)

	if got := h.State(); got != upstreamRecovering {
		t.Fatalf("state=%s, want recovering", got)
	}
	h.mu.Lock()
	gotDeadline := h.recoveryDeadline
	h.mu.Unlock()
	if !gotDeadline.Equal(deadline) {
		t.Fatalf("recovery deadline was extended: got %s, want %s", gotDeadline, deadline)
	}
}

func TestUpstreamHealth_CancelledWaitDoesNotPoisonRecovery(t *testing.T) {
	h := newUpstreamHealth("http://unused", testHealthConfig(), nil)
	h.recordSuccess("test")
	h.recordFailure("test_failure", false)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := h.WaitReady(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("WaitReady error=%v, want context.Canceled", err)
	}
	if got := h.State(); got != upstreamRecovering {
		t.Fatalf("cancelled caller changed shared state to %s", got)
	}
}

func TestUpstreamHealth_BackgroundProbeRecoversKnownDown(t *testing.T) {
	var ready atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if !ready.Load() {
			http.Error(w, "offline", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"model"}]}`))
	}))
	defer server.Close()

	h := newUpstreamHealth(server.URL, testHealthConfig(), server.Client())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)

	waitForHealthState(t, h, upstreamRecovering, 100*time.Millisecond)
	ready.Store(true)
	waitForHealthState(t, h, upstreamHealthy, 100*time.Millisecond)
}

func waitForHealthState(t *testing.T, h *upstreamHealth, want upstreamState, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if h.State() == want {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("state=%s, want %s within %s", h.State(), want, timeout)
}
