package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

// ErrUpstreamUnavailable is returned when the shared LLM circuit is open.
var ErrUpstreamUnavailable = errors.New("inference upstream is unavailable")

type upstreamState string

const (
	upstreamUnknown    upstreamState = "unknown"
	upstreamHealthy    upstreamState = "healthy"
	upstreamRecovering upstreamState = "recovering"
	upstreamDown       upstreamState = "down"
)

// upstreamHealth owns one recovery budget for the LLM upstream. Request
// contexts never drive its probes, so a cancelled caller cannot poison shared
// health state.
type upstreamHealth struct {
	mu sync.Mutex

	upstream string
	client   *http.Client
	cfg      RetryConfig

	state              upstreamState
	lastSuccess        time.Time
	recoveryStarted    time.Time
	recoveryDeadline   time.Time
	recoveryAttempted  bool
	readinessSuccesses int
	changed            chan struct{}
}

func newUpstreamHealth(upstream string, cfg RetryConfig, client *http.Client) *upstreamHealth {
	cfg.applyDefaults()
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}
	return &upstreamHealth{
		upstream: upstream,
		client:   client,
		cfg:      cfg,
		state:    upstreamUnknown,
		changed:  make(chan struct{}),
	}
}

func (h *upstreamHealth) State() upstreamState {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.state
}

func (h *upstreamHealth) Ready() bool {
	return h.State() == upstreamHealthy
}

// Accepting reports whether a request may be forwarded or held through an
// already active restart window. Unknown and known-down states fail fast.
func (h *upstreamHealth) Accepting() bool {
	s := h.State()
	return s == upstreamHealthy || s == upstreamRecovering
}

func (h *upstreamHealth) RetryAfterSeconds() int {
	seconds := int((h.cfg.DownProbeInterval + time.Second - 1) / time.Second)
	if seconds < 1 {
		return 1
	}
	return seconds
}

// WaitReady blocks requests that joined a shared recovery episode. It never
// starts or extends an episode itself.
func (h *upstreamHealth) WaitReady(ctx context.Context) error {
	for {
		h.mu.Lock()
		s := h.state
		changed := h.changed
		h.mu.Unlock()

		switch s {
		case upstreamHealthy:
			return nil
		case upstreamUnknown, upstreamDown:
			return ErrUpstreamUnavailable
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-changed:
		}
	}
}

func (h *upstreamHealth) recordSuccess(source string) {
	h.mu.Lock()
	old := h.state
	downtime := time.Duration(0)
	if !h.recoveryStarted.IsZero() {
		downtime = time.Since(h.recoveryStarted)
	}
	h.state = upstreamHealthy
	h.lastSuccess = time.Now()

	// The first readiness success wakes held requests but does not reset the
	// outage clock. If the real request still gets 503, it rejoins the same
	// episode. A successful real request clears immediately; otherwise two
	// consecutive readiness probes establish stable recovery.
	if source == "readiness" && old == upstreamRecovering {
		h.readinessSuccesses = 1
	} else if source == "readiness" && old == upstreamHealthy && h.recoveryAttempted {
		h.readinessSuccesses++
		if h.readinessSuccesses >= 2 {
			h.clearRecoveryLocked()
		}
	} else {
		h.clearRecoveryLocked()
	}
	h.notifyLocked()
	h.mu.Unlock()

	if old != upstreamHealthy {
		log.Printf("level=info msg=%q from=%s to=%s source=%s downtime=%s",
			"inference upstream state changed", old, upstreamHealthy, source, downtime.Round(time.Second))
	}
}

func (h *upstreamHealth) clearRecoveryLocked() {
	h.recoveryStarted = time.Time{}
	h.recoveryDeadline = time.Time{}
	h.recoveryAttempted = false
	h.readinessSuccesses = 0
}

// recordFailure transitions a recently healthy upstream into one shared
// recovery episode. A cold HTTP 5xx is allowed one episode too, which covers a
// router that is listening while its workers are still loading. Connection
// refusal from a never-healthy upstream remains a fast-fail down state.
func (h *upstreamHealth) recordFailure(source string, startupSignal bool) {
	now := time.Now()
	h.mu.Lock()
	old := h.state
	h.readinessSuccesses = 0

	switch h.state {
	case upstreamHealthy:
		if h.recoveryAttempted {
			if now.Before(h.recoveryDeadline) {
				h.state = upstreamRecovering
			} else {
				h.state = upstreamDown
			}
		} else if !h.lastSuccess.IsZero() && now.Sub(h.lastSuccess) <= h.cfg.RecentSuccessWindow {
			h.startRecoveryLocked(now)
		} else {
			h.state = upstreamDown
			h.recoveryAttempted = true
		}
	case upstreamUnknown:
		if startupSignal && !h.recoveryAttempted {
			h.startRecoveryLocked(now)
		} else {
			h.state = upstreamDown
		}
	case upstreamDown:
		if startupSignal && !h.recoveryAttempted {
			h.startRecoveryLocked(now)
		}
	case upstreamRecovering:
		if !h.recoveryDeadline.IsZero() && !now.Before(h.recoveryDeadline) {
			h.state = upstreamDown
		}
	}

	changed := old != h.state
	if changed {
		h.notifyLocked()
	}
	newState := h.state
	deadline := h.recoveryDeadline
	h.mu.Unlock()

	if changed {
		log.Printf("level=warn msg=%q from=%s to=%s source=%s recovery_deadline=%s",
			"inference upstream state changed", old, newState, source, deadline.Format(time.RFC3339))
	}
}

func (h *upstreamHealth) startRecoveryLocked(now time.Time) {
	h.state = upstreamRecovering
	h.recoveryStarted = now
	h.recoveryDeadline = now.Add(h.cfg.RecoveryTimeout)
	h.recoveryAttempted = true
}

func (h *upstreamHealth) notifyLocked() {
	close(h.changed)
	h.changed = make(chan struct{})
}

// Run probes readiness independently of request lifetimes. A successful
// /v1/models response must contain at least one model, which verifies worker
// readiness rather than merely proving that a router process is listening.
func (h *upstreamHealth) Run(ctx context.Context) {
	for {
		h.probe(ctx)

		interval := h.probeInterval()
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}

func (h *upstreamHealth) probeInterval() time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()

	interval := h.cfg.DownProbeInterval
	switch h.state {
	case upstreamHealthy:
		interval = h.cfg.HealthyProbeInterval
	case upstreamRecovering:
		interval = h.cfg.RecoveryProbeInterval
		remaining := time.Until(h.recoveryDeadline)
		if remaining <= 0 {
			return time.Millisecond
		}
		if remaining < interval {
			interval = remaining
		}
	}
	return interval
}

func (h *upstreamHealth) probe(parent context.Context) {
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.upstream+"/v1/models", nil)
	if err != nil {
		h.recordFailure("readiness_request", false)
		return
	}
	resp, err := h.client.Do(req)
	if err != nil {
		h.recordFailure("readiness_transport", false)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		h.recordFailure(fmt.Sprintf("readiness_http_%d", resp.StatusCode), isStartupStatus(resp.StatusCode))
		return
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil || len(payload.Data) == 0 {
		h.recordFailure("readiness_invalid_models", false)
		return
	}
	h.recordSuccess("readiness")
}

func isStartupStatus(status int) bool {
	return status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}
