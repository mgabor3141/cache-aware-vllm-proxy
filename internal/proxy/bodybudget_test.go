package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"
)

// imageThreadBody builds a body shaped like the traffic that motivated this file: a
// little text and one large base64 blob per image. The 34 MB target is the size the
// live proxy rejected on 2026-09-28, not a round number I picked.
func imageThreadBody(t *testing.T, bytesTarget int) []byte {
	t.Helper()
	blob := strings.Repeat("QUJDRA==", bytesTarget/8)
	payload := map[string]any{
		"model": "default",
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "what is in these?"},
				map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": "data:image/png;base64," + blob,
				}},
			},
		}},
		"max_tokens": 512,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("build body: %v", err)
	}
	return body
}

// TestDecodeAmplification measures how much more than the body size the buffered
// path puts on the heap, because the memory budget is only honest if its weight is.
// Decoding into a map copies every string out of the raw bytes and re-encodes the
// whole thing, so the peak is the raw bytes plus the map plus the marshalled copy.
// The number this prints is what defaultBodyAmplification is set from.
func TestDecodeAmplification(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector inflates allocations")
	}
	raw := imageThreadBody(t, 16<<20)
	var before, mid, after runtime.MemStats
	runtime.GC()
	// No collection during the measurement: a GC between the reads would free the
	// garbage it is meant to count, and once read 0x.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.ReadMemStats(&before)

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	runtime.ReadMemStats(&mid)
	out, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	runtime.ReadMemStats(&after)

	// raw was allocated before the first read, so add it: the budget charges the body's
	// own bytes too.
	n := float64(len(raw))
	decode := (float64(mid.HeapAlloc-before.HeapAlloc) + n) / n
	both := (float64(after.HeapAlloc-before.HeapAlloc) + n) / n
	t.Logf("body=%d bytes  raw+Unmarshal=%.2fx  raw+Unmarshal+Marshal (all allocations, nothing freed)=%.2fx",
		len(raw), decode, both)
	if len(out) == 0 {
		t.Fatalf("marshal produced nothing")
	}
	if both < 1 {
		t.Errorf("measured amplification %.2fx is below 1x, which cannot be right; "+
			"the budget weight derived from it would under-count memory", both)
	}
	if both > float64(defaultBodyAmplification) {
		t.Errorf("measured amplification %.2fx exceeds defaultBodyAmplification=%d; "+
			"the budget under-counts and the default needs raising",
			both, defaultBodyAmplification)
	}
}

func TestByteBudgetWaitsThenAdmitsAfterRelease(t *testing.T) {
	b := newByteBudget(100, 1, 0)
	release, err := b.acquire(context.Background(), 60)
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	if got := b.inUse(); got != 60 {
		t.Fatalf("inUse = %d, want 60", got)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	if _, err := b.acquire(ctx, 60); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second acquire while full = %v, want deadline exceeded", err)
	}

	release()
	release() // idempotent: a deferred release plus an explicit one must not double-pay
	if got := b.inUse(); got != 0 {
		t.Fatalf("inUse after double release = %d, want 0", got)
	}
	release2, err := b.acquire(context.Background(), 60)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	release2()
}

func TestByteBudgetRejectsBodyBiggerThanTheWholeBudget(t *testing.T) {
	b := newByteBudget(100, 1, 0)
	_, err := b.acquire(context.Background(), 200)
	if !errors.Is(err, errBodyOverBudget) {
		t.Fatalf("acquire(200) with limit 100 = %v, want errBodyOverBudget", err)
	}
	// Even with nothing else outstanding. The caller cannot fix this by waiting.
	if got := b.inUse(); got != 0 {
		t.Errorf("rejected acquire changed inUse to %d, want 0", got)
	}
}

func TestByteBudgetChargesWeightAndIgnoresWhenDisabled(t *testing.T) {
	b := newByteBudget(100, 4, 0)
	release, err := b.acquire(context.Background(), 20)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if got := b.inUse(); got != 80 {
		t.Errorf("inUse = %d, want 80 (20 bytes at weight 4)", got)
	}
	release()

	off := newByteBudget(0, 4, 0)
	r2, err := off.acquire(context.Background(), 1<<40)
	if err != nil {
		t.Fatalf("disabled budget refused a huge body: %v", err)
	}
	r2()
	if got := off.inUse(); got != 0 {
		t.Errorf("disabled budget tracked %d bytes", got)
	}
}

func TestByteBudgetChargesUnknownLength(t *testing.T) {
	b := newByteBudget(1<<30, 1, 0)
	release, err := b.acquire(context.Background(), -1)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if got := b.inUse(); got != defaultChunkedBodyBytes {
		t.Errorf("inUse = %d, want the full %d byte chunked charge. An unknown-length body must be charged what it is capped at, or omitting Content-Length becomes a way past the budget",
			got, defaultChunkedBodyBytes)
	}
	release()
}

func TestReadBodyCapsUndeclaredLength(t *testing.T) {
	r := newRequestWithBody(strings.NewReader(strings.Repeat("y", 4096)), -1)
	_, err := readBody(r, 0, 1024)
	if !errors.Is(err, errBodyOverBudget) {
		t.Fatalf("readBody with no declared length past the cap = %v, want errBodyOverBudget", err)
	}
}

func TestReadBodyAcceptsUndeclaredLengthWithinCap(t *testing.T) {
	r := newRequestWithBody(strings.NewReader("abc"), -1)
	raw, err := readBody(r, 0, 1024)
	if err != nil {
		t.Fatalf("readBody: %v", err)
	}
	if string(raw) != "abc" {
		t.Errorf("body = %q, want %q", raw, "abc")
	}
}

func TestReadBodyUsesDeclaredLength(t *testing.T) {
	r := newRequestWithBody(strings.NewReader("hello world"), 11)
	raw, err := readBody(r, 0, 0)
	if err != nil {
		t.Fatalf("readBody: %v", err)
	}
	if string(raw) != "hello world" {
		t.Errorf("body = %q, want %q", raw, "hello world")
	}
}

func TestReadBodyRefusesDeclaredOversizeBeforeReading(t *testing.T) {
	src := &trackedReader{Reader: strings.NewReader(strings.Repeat("x", 4096))}
	r := newRequestWithBody(src, 4096)
	_, err := readBody(r, 1024, 0)
	if !errors.Is(err, errBodyOverBudget) {
		t.Fatalf("readBody = %v, want errBodyOverBudget", err)
	}
	if src.read {
		t.Error("the body was read anyway; a rejection should not make the client upload the payload first")
	}
}

func TestReadBodyShortBodyIsNotAnError(t *testing.T) {
	// A lying ContentLength above the real body must not turn into a hard failure:
	// the payload that did arrive is still forwarded so the engine can judge it.
	r := newRequestWithBody(strings.NewReader("abc"), 500)
	raw, err := readBody(r, 0, 0)
	if err != nil {
		t.Fatalf("readBody on a short body = %v, want the bytes read", err)
	}
	if string(raw) != "abc" {
		t.Errorf("body = %q, want %q", raw, "abc")
	}
}

func TestReadBodyUnboundedUnknownLength(t *testing.T) {
	r := newRequestWithBody(strings.NewReader("chunked payload"), -1)
	r.ContentLength = -1
	raw, err := readBody(r, 0, 0)
	if err != nil {
		t.Fatalf("readBody: %v", err)
	}
	if string(raw) != "chunked payload" {
		t.Errorf("body = %q", raw)
	}
}

type trackedReader struct {
	io.Reader
	read bool
}

func (t *trackedReader) Read(p []byte) (int, error) {
	t.read = true
	return t.Reader.Read(p)
}
func (t *trackedReader) Close() error { return nil }

// newRequestWithBody builds a real *http.Request so readBody is exercised against
// the type it actually receives, with the declared length set the way Go's server
// would set it from the wire.
func newRequestWithBody(body io.Reader, contentLength int64) *http.Request {
	r := httptest.NewRequest("POST", "/v1/chat/completions", body)
	r.ContentLength = contentLength
	return r
}

// The response tail ring belongs with the memory bounds: it is the same idea
// pointed the other way. The old code kept the first 4 MiB of every response and
// dropped the copy past that, which threw away the usage object on long agentic
// turns. Keeping the last bytes instead is both cheaper and correct, since usage is
// written last. This test is the regression that says so.
func TestCapturingWriterKeepsTheEndOfLongResponses(t *testing.T) {
	rec := httptest.NewRecorder()
	c := &capturingWriter{ResponseWriter: rec, status: 200, captureBody: true}
	head := `{"choices":[{"index":0,"message":{"role":"assistant","content":"`
	middle := strings.Repeat("x", 3*maxUsageTailBytes)
	usage := `"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120000,"total_tokens":120456,"completion_tokens":456,"prompt_tokens_details":{"cached_tokens":119040}}}`

	for _, chunk := range []string{head, middle, usage} {
		if _, err := c.Write([]byte(chunk)); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	if got, want := rec.Body.Len(), len(head)+len(middle)+len(usage); got != want {
		t.Errorf("client received %d bytes, want %d: capture must never withhold bytes", got, want)
	}
	if len(c.usageTail()) > maxUsageTailBytes {
		t.Errorf("kept %d bytes of tail, want at most %d", len(c.usageTail()), maxUsageTailBytes)
	}
	if !c.truncated {
		t.Error("truncated not set on a response larger than the tail window")
	}
	u, ok := parseUsageFromTail(c.usageTail())
	if !ok {
		t.Fatalf("usage lost on a response whose middle was discarded; this is the regression this test exists to catch")
	}
	if u.Prompt != 120000 || u.Cached != 119040 {
		t.Errorf("parsed %+v, want prompt 120000 cached 119040", u)
	}
}
