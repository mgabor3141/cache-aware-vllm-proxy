// retry.go: a RoundTripper that holds requests through an engine restart instead of
// failing them.
//
// A request that fails with a transient error waits on the shared upstream health state
// (health.go) and is retried once the engine answers its readiness probe again. All
// waiting requests share one recovery clock (recovery_timeout), so an outage costs each
// of them at most that, not an independent retry budget each.
//
// Retried: network errors before response headers (connection refused, reset, EOF, dial
// timeout), and the statuses in retry_status_codes (default 502, 503, 504: an engine
// shutting down or not yet serving).
//
// Not retried: 2xx-4xx; errors after response headers (bytes already reached the client,
// so a retry would duplicate them: a stream cut mid-way stays cut); a cancelled request.
//
// Bodies are replayed through req.GetBody, which the proxy sets when it buffers a body.
package proxy

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// RetryConfig controls how requests ride out an engine outage: which statuses count as
// transient, and the shared recovery clock and readiness probes (health.go).
type RetryConfig struct {
	// RetryStatusCodes lists HTTP status codes that trigger a retry.
	// Default {502, 503, 504} covers the bad-gateway / service-
	// unavailable / gateway-timeout family that upstream usually
	// produces during graceful shutdown.
	RetryStatusCodes []int `yaml:"retry_status_codes"`

	// RecentSuccessWindow is how recently readiness must have succeeded
	// for a failure to open a long recovery episode. It is deliberately
	// longer than HealthyProbeInterval so one delayed probe is harmless.
	RecentSuccessWindow time.Duration `yaml:"recent_success_window"`

	// RecoveryTimeout is the single shared retry budget for an outage.
	// New requests cannot extend it.
	RecoveryTimeout time.Duration `yaml:"recovery_timeout"`

	// Probe intervals control the independent /v1/models readiness loop.
	HealthyProbeInterval  time.Duration `yaml:"healthy_probe_interval"`
	RecoveryProbeInterval time.Duration `yaml:"recovery_probe_interval"`
	DownProbeInterval     time.Duration `yaml:"down_probe_interval"`
}

func (c *RetryConfig) applyDefaults() {
	if len(c.RetryStatusCodes) == 0 {
		c.RetryStatusCodes = []int{
			http.StatusBadGateway,
			http.StatusServiceUnavailable,
			http.StatusGatewayTimeout,
		}
	}
	if c.RecentSuccessWindow <= 0 {
		c.RecentSuccessWindow = 2 * time.Minute
	}
	if c.RecoveryTimeout <= 0 {
		c.RecoveryTimeout = 8 * time.Minute
	}
	if c.HealthyProbeInterval <= 0 {
		c.HealthyProbeInterval = 30 * time.Second
	}
	if c.RecoveryProbeInterval <= 0 {
		c.RecoveryProbeInterval = 3 * time.Second
	}
	if c.DownProbeInterval <= 0 {
		c.DownProbeInterval = 10 * time.Second
	}
}

// retryingTransport wraps a base RoundTripper with retry-on-transient.
// Goroutine-safe: holds no mutable state per call (each RoundTrip
// computes its own backoff schedule from cfg).
type retryingTransport struct {
	base   http.RoundTripper
	cfg    RetryConfig
	health *upstreamHealth
}

func newHealthAwareRetryingTransport(base http.RoundTripper, cfg RetryConfig, health *upstreamHealth) *retryingTransport {
	if base == nil {
		base = http.DefaultTransport
	}
	cfg.applyDefaults()
	return &retryingTransport{base: base, cfg: cfg, health: health}
}

// RoundTrip implements http.RoundTripper: it waits on the shared outage state
// instead of giving every request an independent retry clock. The recovery deadline in
// upstreamHealth bounds the loop.
func (t *retryingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var lastErr error
	for attempt := 1; ; attempt++ {
		if err := t.health.WaitReady(req.Context()); err != nil {
			if lastErr != nil && errors.Is(err, ErrUpstreamUnavailable) {
				return nil, fmt.Errorf("%w: %v", ErrUpstreamUnavailable, lastErr)
			}
			return nil, err
		}

		if attempt > 1 && req.Body != nil {
			if req.GetBody == nil {
				return nil, fmt.Errorf(
					"retry: request has body but no GetBody; previous error: %w", lastErr)
			}
			newBody, err := req.GetBody()
			if err != nil {
				return nil, fmt.Errorf("retry: GetBody: %w", err)
			}
			req.Body = newBody
		}

		resp, err := t.base.RoundTrip(req)
		if err == nil {
			if !t.shouldRetryStatus(resp.StatusCode) {
				t.health.recordSuccess("request")
				if attempt > 1 {
					log.Printf("level=info msg=%q attempt=%d status=%d path=%s",
						"proxy recovery retry succeeded", attempt, resp.StatusCode, req.URL.Path)
				}
				return resp, nil
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("upstream returned %d", resp.StatusCode)
			t.health.recordFailure(fmt.Sprintf("request_http_%d", resp.StatusCode), true)
			continue
		}

		if !isRetryableError(err) {
			return nil, err
		}
		lastErr = err
		t.health.recordFailure("request_transport", false)
	}
}

// shouldRetryStatus reports whether the given HTTP status code is in
// the configured retry-status list.
func (t *retryingTransport) shouldRetryStatus(code int) bool {
	for _, c := range t.cfg.RetryStatusCodes {
		if c == code {
			return true
		}
	}
	return false
}

// isRetryableError decides whether a transport-level error is worth
// retrying. We retry on TCP-layer errors that typically resolve
// themselves once the upstream comes back: connect refused, broken
// pipe, EOF on a half-open connection, dial timeout.
//
// We do NOT retry context errors (caller gave up) or errors that
// indicate a misconfiguration on our side (a hostname that doesn't
// resolve at all, for example, won't fix itself).
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	msg := err.Error()
	for _, pat := range []string{
		"connection refused",
		"connection reset",
		"broken pipe",
		"no route to host",
		"i/o timeout",
		"unexpected EOF",
		"EOF",
		"server closed connection",
		// httputil-style wrapper around the above sometimes hides the
		// inner cause behind a generic "http: server closed idle
		// connection" or similar; this catches those.
		"http: server",
	} {
		if strings.Contains(msg, pat) {
			return true
		}
	}
	return false
}
