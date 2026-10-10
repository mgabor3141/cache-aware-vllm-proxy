// Structured request logging.
//
// One Event per completed request, marshalled as JSON, written to
// stdout, ready for any log shipper (e.g. Loki: `{container="cavp"} | json`).
//
// Token counts (prompt/completion/total) are captured when available:
// from the JSON `usage` object on non-streaming responses, and from the
// final usage SSE chunk on streaming ones (rewriteBody injects
// stream_options.include_usage set-if-missing so that chunk is present).
// They are omitempty, so requests without a usable count (errors,
// /v1/models, oversized non-streaming bodies, or a caller who set
// include_usage:false) simply omit them. See usage.go.
package proxy

import (
	"encoding/json"
	"io"
	"sync"
)

// Event is the wire format of one log line.
//
// Field order matches roughly the lifecycle (when each becomes known):
// timestamp first, then the request shape, then the auth identity,
// then the alias/target/priority decisions, finally the outcome.
// JSON has no field-order guarantees but humans reading raw logs
// appreciate it.
type Event struct {
	TS           string `json:"ts"`
	Method       string `json:"method"`
	Path         string `json:"path"`
	User         string `json:"user,omitempty"`     // empty on auth failure
	Auth         string `json:"auth,omitempty"`     // "key" or "oidc"; empty on auth failure
	Sub          string `json:"sub,omitempty"`      // OIDC sub claim (raw UUID); empty for static-key auth
	Alias        string `json:"alias,omitempty"`    // model name the caller requested
	Target       string `json:"target,omitempty"`   // model name forwarded to vLLM
	Priority     int    `json:"priority,omitempty"` // resolved priority (omitted when 0)
	Status       int    `json:"status"`
	LatencyMS    int64  `json:"latency_ms"`
	QueueMS      int64  `json:"queue_ms,omitempty"`
	RequestBytes int64  `json:"request_bytes,omitempty"`
	// ContentLength and Error are set when the request body could not be read
	// or parsed, so a failed upload shows how much arrived against what the
	// client declared (a reverse proxy's read timeout, an aborted upload).
	ContentLength   int64  `json:"content_length,omitempty"`
	Error           string `json:"error,omitempty"`
	ActiveOnArrival int    `json:"active_on_arrival,omitempty"`
	QueuedOnArrival int    `json:"queued_on_arrival,omitempty"`
	// Cache-aware admission (cache.go). EstPromptTokens is the prompt size admission
	// used (exact when Tokenized); PredictedCachedTokens what the KV mirror said was
	// cached at admission, to compare with cached_prompt_tokens; ChargeTokens the space
	// reserved (prompt plus expected output); PrefillCostS its own GPU seconds;
	// DisplacedTokens and DisplaceCostS what it took from idle sessions' holds.
	CacheUsedOnArrival    int     `json:"cache_used_on_arrival,omitempty"`
	Tokenized             bool    `json:"tokenized,omitempty"`
	TokenizeMS            int64   `json:"tokenize_ms,omitempty"`
	TokenizeError         string  `json:"tokenize_error,omitempty"`
	EstPromptTokens       int     `json:"est_prompt_tokens,omitempty"`
	PredictedCachedTokens int     `json:"predicted_cached_tokens,omitempty"`
	ChargeTokens          int     `json:"charge_tokens,omitempty"`
	PrefillCostS          float64 `json:"prefill_cost_s,omitempty"`
	DisplacedTokens       int     `json:"displaced_tokens,omitempty"`
	DisplaceCostS         float64 `json:"displace_cost_s,omitempty"`
	maxTokens             int
	textBytes             int64
	admitBody             map[string]any
	generated             *OutputCounter // output tokens streamed so far
	report                *AdmissionReport
	// Hashed conversation identity (X-Session-Key, else a fingerprint of the opening
	// and the caller). With cache-aware admission it keys the session's cache hold.
	Session string `json:"session,omitempty"`
	// Token usage, when known. omitempty: absent on errors,
	// /v1/models, and any response we couldn't read a usage object from.
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
	// CachedPromptTokens is the prefix-cache hit for this one request, taken from
	// usage.prompt_tokens_details.cached_tokens. Paired with Session it answers the
	// question the aggregate engine counters cannot: how much of a returning agent
	// turn had to be recomputed, and how long that session had been idle.
	//
	// The comment this field used to carry said omission means "the engine reported no
	// details", and that was wrong in both directions. omitempty drops a zero value, so a
	// real cache miss is omitted too, and a request with no usage object at all omits it as
	// well. Anyone reading the log could not tell a miss from a blind spot, which is how a
	// 97.6% reuse rate got reported from data that supports 86.4% at worst-case-blind-spot
	// and 58.5% if every blind spot were a miss. Reading this field now requires
	// UsageDetails:
	//
	//   usage_details true  + cached_prompt_tokens present   hit, that many tokens
	//   usage_details true  + field absent                   genuine full miss, cached was 0
	//   usage_details false + field absent                   engine reported nothing, unknown
	//
	// Keeping omitempty on the number is deliberate. Without it every /v1/models poll and
	// every request without usage would emit cached_prompt_tokens:0, which would read as a miss and
	// invent the very ambiguity this removes.
	CachedPromptTokens int  `json:"cached_prompt_tokens,omitempty"`
	UsageDetails       bool `json:"usage_details,omitempty"`
	// UsageTruncated marks a usage figure read from a response whose middle was
	// discarded: only the tail is kept, and a response larger than that is flagged
	// here rather than silently reported as if it had been read whole.
	UsageTruncated bool `json:"usage_truncated,omitempty"`
	// BodyReserveMS is time spent waiting for the in-flight body memory budget.
	// Zero means the body was immediately affordable; non-zero says this request
	// was held back because other requests were holding their bodies.
	BodyReserveMS int64 `json:"body_reserve_ms,omitempty"`
}

// EventLogger is a thin shim around an io.Writer with a mutex.
// Concurrent goroutines (one per in-flight request) all log through
// here; the mutex keeps lines from interleaving even if the underlying
// Writer isn't atomic per Write.
type EventLogger struct {
	mu sync.Mutex
	w  io.Writer
}

func NewEventLogger(w io.Writer) *EventLogger {
	return &EventLogger{w: w}
}

func (l *EventLogger) Log(e Event) {
	data, err := json.Marshal(e)
	if err != nil {
		// Encoding a struct of basic types shouldn't fail; if it does
		// we'd rather drop the line than crash the proxy.
		return
	}
	data = append(data, '\n')
	l.mu.Lock()
	_, _ = l.w.Write(data)
	l.mu.Unlock()
}
