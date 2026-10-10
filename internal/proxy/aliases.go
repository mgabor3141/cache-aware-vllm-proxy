// Dynamic alias resolution.
//
// User-facing model names like "default" map to whatever the upstream
// vLLM is currently serving, queried from /v1/models and cached. This
// keeps the proxy config oblivious to model swaps: deploy a new model,
// the proxy picks it up at the next refresh.
package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"
)

// ModelResolver maintains a cached view of the upstream model list.
// The atomic.Pointer makes reads lock-free in the hot path; the
// resolver goroutine writes whenever it successfully refreshes.
type ModelResolver struct {
	upstream    string
	client      *http.Client
	current     atomic.Pointer[string] // served model id
	maxModelLen atomic.Int64           // upstream context ceiling, zero when unavailable
}

func NewModelResolver(upstream string, client *http.Client) *ModelResolver {
	return &ModelResolver{upstream: upstream, client: client}
}

// Current returns the cached model id, or "" if we haven't successfully
// refreshed yet. Callers can decide how to handle the empty case
// (typically: return 503 to the client since the auto-alias can't be
// resolved without it).
func (r *ModelResolver) Current() string {
	if p := r.current.Load(); p != nil {
		return *p
	}
	return ""
}

// MaxModelLen returns the context ceiling reported by the current upstream.
// Zero means the upstream omitted it or no refresh has completed yet.
func (r *ModelResolver) MaxModelLen() int64 {
	return r.maxModelLen.Load()
}

// Refresh queries /v1/models once. Errors are returned, not logged
// here, so the caller decides whether to be noisy or quiet.
func (r *ModelResolver) Refresh(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.upstream+"/v1/models", nil)
	if err != nil {
		return err
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("upstream /v1/models returned %d", resp.StatusCode)
	}
	var payload struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int64  `json:"max_model_len"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return fmt.Errorf("decode /v1/models: %w", err)
	}
	if len(payload.Data) == 0 {
		return fmt.Errorf("upstream /v1/models returned empty list")
	}
	// We assume one model. If we ever serve more than one, the alias
	// config would need explicit targets (not "<auto>"); the empty
	// assumption here would be a deliberate decision then.
	id := payload.Data[0].ID
	r.maxModelLen.Store(payload.Data[0].MaxModelLen)
	r.current.Store(&id)
	return nil
}

// RunRefreshLoop refreshes once immediately, then on `interval`.
// Blocks until ctx is cancelled. Errors during refresh are reported
// via the cb so the caller controls log format (we don't import the
// logger here on purpose; keeps this file dependency-free for tests).
func (r *ModelResolver) RunRefreshLoop(ctx context.Context, interval time.Duration, cb func(error)) {
	doIt := func() {
		// Bound each refresh well under the loop interval so a stuck
		// upstream doesn't block us indefinitely between attempts.
		c, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		if err := r.Refresh(c); err != nil && cb != nil {
			cb(err)
		}
	}
	doIt()
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			doIt()
		}
	}
}

// ResolveAlias takes a user-requested model name and returns:
//
//	target: the actual model id to send to vLLM
//	override: the priority override the alias enforces (or nil)
//	known: whether the name matched an alias entry
//
// If known is false the caller is expected to pass through the name
// unchanged (a request for the literal served model id).
// The "<auto>" sentinel is resolved here; an empty current value
// returns ("", nil, true) and the caller must surface that as a
// retryable error to the client.
func ResolveAlias(name string, aliases map[string]AliasEntry, r *ModelResolver) (target string, override *int, known bool) {
	a, ok := aliases[name]
	if !ok {
		return name, nil, false
	}
	target = a.Target
	if target == AutoTarget {
		target = r.Current()
	}
	return target, a.PriorityOverride, true
}
