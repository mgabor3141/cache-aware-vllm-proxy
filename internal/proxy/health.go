package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// readiness tracks whether the engine can serve: GET /v1/models answers 200 and lists
// a model. A probe runs every interval, independent of requests. While the engine is
// not ready, admitted requests wait for it (Wait) instead of failing, however long
// that takes; a client that gives up simply leaves. Nothing is retried: a request that
// fails once it reached the engine returns its error.
type readiness struct {
	url      string
	client   *http.Client
	interval time.Duration

	mu      sync.Mutex
	ready   bool
	changed chan struct{} // closed and replaced on every change of ready

	waiting atomic.Int64 // requests in Wait while the engine is not ready
}

func newReadiness(upstream string, interval time.Duration) *readiness {
	return &readiness{
		url:      upstream + "/v1/models",
		client:   &http.Client{Timeout: 5 * time.Second},
		interval: interval,
		changed:  make(chan struct{}),
	}
}

// Ready reports the last probe's result.
func (h *readiness) Ready() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.ready
}

// Wait returns nil once the engine is ready, or ctx's error if it is cancelled first.
func (h *readiness) Wait(ctx context.Context) error {
	counted := false
	for {
		h.mu.Lock()
		ready, changed := h.ready, h.changed
		h.mu.Unlock()
		if ready {
			return nil
		}
		if !counted {
			counted = true
			h.waiting.Add(1)
			defer h.waiting.Add(-1)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-changed:
		}
	}
}

func (h *readiness) set(ready bool, why string) {
	h.mu.Lock()
	if h.ready == ready {
		h.mu.Unlock()
		return
	}
	h.ready = ready
	close(h.changed)
	h.changed = make(chan struct{})
	h.mu.Unlock()
	if ready {
		log.Printf("level=info msg=\"engine ready\"")
	} else {
		log.Printf("level=warn msg=\"engine not ready; admitted requests wait\" reason=%q", why)
	}
}

// Run probes until ctx is cancelled, starting at once.
func (h *readiness) Run(ctx context.Context) {
	t := time.NewTicker(h.interval)
	defer t.Stop()
	for {
		err := h.probe(ctx)
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			h.set(false, err.Error())
		} else {
			h.set(true, "")
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (h *readiness) probe(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.url, nil)
	if err != nil {
		return err
	}
	resp, err := h.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("/v1/models: HTTP %d", resp.StatusCode)
	}
	var models struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&models); err != nil {
		return fmt.Errorf("/v1/models: %w", err)
	}
	if len(models.Data) == 0 {
		return fmt.Errorf("/v1/models lists no model")
	}
	return nil
}
