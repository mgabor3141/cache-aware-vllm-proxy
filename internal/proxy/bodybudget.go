package proxy

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"time"
)

// defaultBodyAmplification is the memory multiple charged per body when the config does
// not set body_amplification. The buffered path holds the raw bytes, the decoded map and
// the re-encoded bytes at once, plus the encoder's growing buffers. With the collector off,
// TestDecodeAmplification counts every allocation from the raw body through the re-encode:
// 5.00x on a 16 MB image-shaped body. That bounds the live peak for this shape; the default
// adds one more for the copy the /tokenize request makes and for encoder changes.
const defaultBodyAmplification = 6

// defaultChunkedBodyBytes bounds a body that arrives with no declared length.
// Without Content-Length there is nothing to reserve against before reading, so an
// unknown-length body is charged, and capped, at this size instead. Every client in
// front of this proxy declares a length; this is the conservative branch for a
// client that does not, and the size that was measured but not promised is reported
// in the event log either way. It is deliberately larger than any single request we
// expect, because the cap is the charge: at the default amplification each such upload
// reserves 768 MiB of the in-flight budget, so they are admitted a few at a time
// rather than ever being unaccounted for.
const defaultChunkedBodyBytes = 128 << 20

// errBodyOverBudget means one request body is larger than the entire in-flight
// budget on its own, so no amount of waiting could serve it. It is reported as
// 503 with a number in it rather than 400, because the caller did nothing wrong:
// the payload is valid and the proxy is simply too small for it right now.
var errBodyOverBudget = errors.New("request body exceeds the proxy in-flight memory budget")

// byteBudget bounds the total request-body bytes the proxy holds in memory at
// once.
//
// The proxy has to read the body to rewrite the model alias, so an unbounded
// per-request size is only safe if the *total* is bounded. Without this, the
// concurrency limit and the body size multiply: admission allows
// max_active requests running and any number queued, and a body read before admission
// stays resident through the whole queue wait, which can be thirty minutes. At
// four active and thirty-two queued, a 300 MB thread is enough to make the
// proxy the largest process on the host, and it has no container memory limit
// of its own.
//
// Cost is charged as the declared length times a decode amplification weight,
// because the buffered path does not hold one copy of the body: it decodes the
// JSON into a map and re-encodes it. The weight is measured, not guessed; see
// TestDecodeAmplification.
type byteBudget struct {
	limit   int64
	weight  int64
	chunked int64
	poll    time.Duration
	mu      sync.Mutex
	used    int64
}

func newByteBudget(limit, weight, chunked int64) *byteBudget {
	if weight < 1 {
		weight = 1
	}
	if chunked <= 0 {
		chunked = defaultChunkedBodyBytes
	}
	return &byteBudget{limit: limit, weight: weight, chunked: chunked, poll: 25 * time.Millisecond}
}

// acquire reserves cost amplified bytes, waiting until they are free. It returns
// a release that is safe to call more than once. A disabled budget (limit <= 0)
// hands back a no-op release so callers do not need their own switch.
func (b *byteBudget) acquire(ctx context.Context, length int64) (func(), error) {
	if b == nil || b.limit <= 0 {
		return func() {}, nil
	}
	cost := length
	if cost <= 0 {
		// Chunked upload with no declared length. Charge the same amount readBody
		// will refuse to exceed, so an anonymous stream cannot bypass the budget by
		// omitting Content-Length.
		cost = b.chunked
	}
	cost *= b.weight
	if cost > b.limit {
		return nil, errBodyOverBudget
	}
	for {
		b.mu.Lock()
		if b.used+cost <= b.limit {
			b.used += cost
			b.mu.Unlock()
			var once sync.Once
			return func() {
				once.Do(func() {
					b.mu.Lock()
					b.used -= cost
					b.mu.Unlock()
				})
			}, nil
		}
		b.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(b.poll):
		}
	}
}

func (b *byteBudget) inUse() int64 {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.used
}

// readBody reads a request body with one allocation.
//
// When the length is known, the exact number of bytes is allocated once and
// filled with io.ReadFull. io.ReadAll on a body of unknown growth instead doubles
// its buffer, so a 34 MB upload transiently reserves roughly twice what it ends
// up needing, and on a 300 MB thread that overshoot is the difference between a
// working proxy and an OOM kill of something unrelated.
//
// limit is the optional per-request cap. A declared length over it is refused
// before any of the body is read, so a rejected upload costs the client nothing
// beyond the headers. That matters: the old code read the whole 34 MB, spent
// 15 to 32 seconds receiving it, and only then returned 400.
// chunkedLimit caps a body with no declared length. Zero means no cap.
func readBody(r *http.Request, limit, chunkedLimit int64) ([]byte, error) {
	if limit > 0 && r.ContentLength > limit {
		return nil, errBodyOverBudget
	}
	if r.ContentLength > 0 {
		n := r.ContentLength
		if limit > 0 && n > limit {
			n = limit
		}
		buf := make([]byte, n)
		read, err := io.ReadFull(r.Body, buf)
		if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
			return buf[:read], err
		}
		return buf[:read], nil
	}
	if limit > 0 {
		raw, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
		if err != nil {
			return raw, err
		}
		if int64(len(raw)) > limit {
			return raw, errBodyOverBudget
		}
		return raw, nil
	}
	if chunkedLimit > 0 {
		raw, err := io.ReadAll(io.LimitReader(r.Body, chunkedLimit+1))
		if err != nil {
			return raw, err
		}
		if int64(len(raw)) > chunkedLimit {
			return raw, errBodyOverBudget
		}
		return raw, nil
	}
	return io.ReadAll(r.Body)
}
