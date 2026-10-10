// Request logging: one JSON line per API request on stdout, written when the request
// ends, whatever the outcome. Operational messages (state changes, warnings) go to
// stderr through the standard logger.
package proxy

import (
	"encoding/json"
	"io"
	"sync"
)

// Status codes that exist only in the log: the client got no response, or not this one.
const (
	// statusClientGone: the client went away before a response started, while waiting
	// for the engine or for admission (nginx's convention).
	statusClientGone = 499
)

// Event is one log line. Fields appear roughly in the order they become known.
type Event struct {
	TS       string `json:"ts"`
	Method   string `json:"method"`
	Path     string `json:"path"`
	User     string `json:"user,omitempty"` // empty when authentication failed
	Auth     string `json:"auth,omitempty"` // auth mode that accepted the caller
	Model    string `json:"model,omitempty"`
	Stream   bool   `json:"stream,omitempty"`
	Priority int    `json:"priority,omitempty"`
	// Status is the outcome: the response status, or for a stream (whose client always
	// gets 200 once it started) the engine's status; 499 when the client left while
	// waiting. Error says what went wrong, including a stream cut short.
	Status       int    `json:"status"`
	Error        string `json:"error,omitempty"`
	LatencyMS    int64  `json:"latency_ms"`
	ReadyWaitMS  int64  `json:"ready_wait_ms,omitempty"` // waiting for the engine to be ready
	QueueMS      int64  `json:"queue_ms,omitempty"`      // waiting for admission
	RequestBytes int64  `json:"request_bytes,omitempty"`
	// ContentLength is the declared body size, logged when the body could not be read,
	// so a cut upload shows how much arrived.
	ContentLength   int64 `json:"content_length,omitempty"`
	ActiveOnArrival int   `json:"active_on_arrival,omitempty"`
	QueuedOnArrival int   `json:"queued_on_arrival,omitempty"`
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
	// Session is the hashed conversation identity (session.go); it keys the session's
	// cache hold.
	Session string `json:"session,omitempty"`
	// Token usage as the engine reported it; absent when it reported none.
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
	// CachedPromptTokens is usage.prompt_tokens_details.cached_tokens. Read it with
	// UsageDetails: details true and the field absent is a real miss (0 cached); details
	// false means the engine reported nothing.
	CachedPromptTokens int  `json:"cached_prompt_tokens,omitempty"`
	UsageDetails       bool `json:"usage_details,omitempty"`

	// Inputs carried from body parsing to admission; not logged.
	maxTokens int
	textBytes int64
	body      map[string]any
	generated *OutputCounter // output tokens streamed so far
	report    *AdmissionReport
}

func (ev *Event) setUsage(u tokenUsage) {
	ev.PromptTokens, ev.CompletionTokens, ev.TotalTokens = u.Prompt, u.Completion, u.Total
	ev.CachedPromptTokens, ev.UsageDetails = u.Cached, u.Details
}

// EventLogger writes events, one line each, safe for concurrent use.
type EventLogger struct {
	mu sync.Mutex
	w  io.Writer
}

func NewEventLogger(w io.Writer) *EventLogger {
	return &EventLogger{w: w}
}

func (l *EventLogger) Log(e *Event) {
	data, err := json.Marshal(e)
	if err != nil {
		return
	}
	data = append(data, '\n')
	l.mu.Lock()
	_, _ = l.w.Write(data)
	l.mu.Unlock()
}
