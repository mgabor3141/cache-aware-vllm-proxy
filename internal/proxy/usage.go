// usage.go — token-usage extraction for request logging.
//
// Two response shapes carry token counts, and we read both:
//
//	Non-streaming: the JSON body has a top-level `usage` object
//	  ({prompt_tokens, completion_tokens, total_tokens}). parseResponseUsage
//	  pulls it out of the captured response body.
//
//	Streaming (SSE): when the request sets stream_options.include_usage,
//	  vLLM appends a final `data: {... "choices":[], "usage":{...}}` chunk
//	  just before `data: [DONE]`. Intermediate chunks carry "usage": null.
//	  usageScanner watches the bytes flowing to the client and keeps the
//	  last non-null usage it sees. rewriteBody injects include_usage
//	  set-if-missing so this chunk is always present.
//
// Both feed the same three fields onto the per-request Event so usage is
// queryable in Loki/Grafana (sum by user, model, etc.).
package proxy

import (
	"bytes"
	"encoding/json"
)

// tokenUsage is the proxy-internal, decoded form.
type tokenUsage struct {
	Prompt int
	// Cached is how many of the prompt tokens the engine found already in its
	// prefix cache. This is the only trustworthy measure of cache reuse we have:
	// the aggregate vllm:prefix_cache_queries_total counter counts per scheduling
	// attempt rather than per request, so under queue pressure its deltas overshoot
	// by roughly 1.3 to 2x, and prefix eviction does not require preemption, so
	// num_preemptions_total can sit at zero while hundreds of thousands of tokens
	// are recomputed. Prompt minus Cached is the recompute bill for one turn.
	Cached     int
	Completion int
	Total      int
	// Details records whether the engine sent prompt_tokens_details at all. It has to be
	// carried separately because Cached cannot express it: a turn with a genuine cache miss
	// has Cached == 0, and a turn whose engine reports no details has Cached == 0 because we
	// never filled it in. On 2026-10-02 that ambiguity cost a conclusion. The proxy log
	// serialises CachedPromptTokens with omitempty, so the JSON line for "engine sent
	// details, the miss was real" is indistinguishable from "engine sent no details": over
	// the 33 hours of access log reviewed, 329 of 792 large turns omit the field, worth
	// 33.8M prompt tokens, and the reuse rate for large turns swings between 86.4% and 58.5%
	// depending on which reading is assumed. Details is the answer to that question, from the
	// only party that can know: the response itself.
	Details bool
}

// promptTokensDetails carries the fields vLLM added under prompt_tokens_details.
// CachedTokens is present on chat completions once the engine has prefix caching
// enabled; it is absent rather than zero on paths that do not report it, which is
// why this is a pointer and why Cached stays 0 rather than being guessed.
type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// usageWire matches the OpenAI/vLLM `usage` object on the wire.
type usageWire struct {
	PromptTokens     int                  `json:"prompt_tokens"`
	PromptDetails    *promptTokensDetails `json:"prompt_tokens_details"`
	CompletionTokens int                  `json:"completion_tokens"`
	TotalTokens      int                  `json:"total_tokens"`
}

func (u usageWire) decoded() tokenUsage {
	out := tokenUsage{Prompt: u.PromptTokens, Completion: u.CompletionTokens, Total: u.TotalTokens}
	if u.PromptDetails != nil {
		out.Cached = u.PromptDetails.CachedTokens
		out.Details = true
	}
	return out
}

// parseUsageFromTail extracts usage from the end of a captured response, which is
// where vLLM puts it: after the choices in a JSON completion, and in the final data
// chunk of a stream.
//
// Two guards keep a tail scan from inventing numbers. The object must sit at the
// end of what was captured, followed only by closing punctuation, because a model
// that writes the literal text "usage":{...} inside a code sample must not become
// our token accounting. And prompt_tokens must be positive, since no real
// completion reports zero prompt tokens. Both guards exist because the previous
// implementation parsed the whole body as JSON and so could not be fooled at all;
// dropping that guarantee has to be paid for explicitly.
func parseUsageFromTail(tail []byte) (tokenUsage, bool) {
	key := []byte(`"usage"`)
	for idx := bytes.LastIndex(tail, key); idx >= 0; idx = bytes.LastIndex(tail[:idx], key) {
		if u, ok := usageObjectAt(tail[idx+len(key):]); ok {
			return u, true
		}
	}
	return tokenUsage{}, false
}

// usageObjectAt parses a usage value from the bytes immediately following a
// `"usage"` key, and only accepts it if nothing but JSON closing punctuation
// follows it.
func usageObjectAt(p []byte) (tokenUsage, bool) {
	p = bytes.TrimLeft(p, " \t\r\n:")
	if len(p) == 0 || p[0] != '{' {
		return tokenUsage{}, false
	}
	depth, inString, escaped := 0, false, false
	for i := 0; i < len(p); i++ {
		c := p[i]
		switch {
		case inString && escaped:
			escaped = false
		case inString && c == '\\':
			escaped = true
		case inString && c == '"':
			inString = false
		case c == '"':
			inString = true
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth > 0 {
				continue
			}
			if !allClosers(p[i+1:]) {
				return tokenUsage{}, false
			}
			var uw usageWire
			if err := json.Unmarshal(p[:i+1], &uw); err != nil {
				return tokenUsage{}, false
			}
			u := uw.decoded()
			if u.Prompt <= 0 {
				return tokenUsage{}, false
			}
			return u, true
		}
	}
	return tokenUsage{}, false
}

// allClosers reports whether b contains only structural closing bytes, the shape a
// document or an SSE chunk ends with.
func allClosers(b []byte) bool {
	for _, c := range b {
		switch c {
		case '}', ']', ' ', '\t', '\r', '\n', '"':
		default:
			return false
		}
	}
	return true
}

// parseResponseUsage extracts usage from a non-streaming completion,
// chat, or embeddings response body. Returns ok=false when the body
// isn't JSON, has no usage object, or reports zero total tokens (which
// we treat as "no usable count" rather than logging a misleading 0).
func parseResponseUsage(body []byte) (tokenUsage, bool) {
	var r struct {
		Usage *usageWire `json:"usage"`
	}
	if err := json.Unmarshal(body, &r); err != nil {
		return tokenUsage{}, false
	}
	if r.Usage == nil || r.Usage.TotalTokens == 0 {
		return tokenUsage{}, false
	}
	return r.Usage.decoded(), true
}

// maxUsagePending caps the bytes usageScanner holds between newlines.
// SSE is newline-delimited, so a complete data line flushes the pending
// buffer; this guard only matters if an upstream ever streamed a single
// enormous line with no newline, in which case we keep just the tail
// (enough to catch a usage chunk) rather than growing without bound.
const maxUsagePending = 1 << 20 // 1 MiB

// usageScanner incrementally scans piped SSE bytes for the final usage
// chunk. It is fed each buffer as it streams to the client (it never
// blocks or alters the stream) and remembers the last non-null usage.
//
// Not safe for concurrent use; in the streaming path only the single
// read-loop goroutine writes to it (the keepalive goroutine writes
// keepalive comments straight to the client, never through here).
type usageScanner struct {
	// chunks, when set, counts the streamed output for admission.
	chunks  *OutputCounter
	pending []byte
	last    tokenUsage
	found   bool
}

// Write feeds streamed bytes to the scanner. It mirrors io.Writer's
// signature for convenience but never errors and never short-writes.
func (s *usageScanner) Write(p []byte) {
	s.pending = append(s.pending, p...)
	for {
		i := bytes.IndexByte(s.pending, '\n')
		if i < 0 {
			break
		}
		line := s.pending[:i]
		s.pending = s.pending[i+1:]
		s.scanLine(line)
	}
	if len(s.pending) > maxUsagePending {
		s.pending = s.pending[len(s.pending)-maxUsagePending:]
	}
}

func (s *usageScanner) scanLine(line []byte) {
	line = bytes.TrimSpace(line)
	if !bytes.HasPrefix(line, []byte("data:")) {
		return
	}
	payload := bytes.TrimSpace(line[len("data:"):])
	choice := s.chunks != nil && bytes.Contains(payload, []byte(`"choices":[{`))
	// Cheap pre-filter: the overwhelming majority of chunks carry
	// "usage": null and we skip the JSON decode for them.
	var chunk struct {
		Usage *usageWire `json:"usage"`
	}
	if bytes.Contains(payload, []byte("\"usage\"")) && json.Unmarshal(payload, &chunk) == nil &&
		chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
		s.last = chunk.Usage.decoded()
		s.found = true
		if s.chunks != nil && chunk.Usage.CompletionTokens > 0 {
			s.chunks.Usage(int64(chunk.Usage.CompletionTokens))
			return
		}
	}
	if choice {
		s.chunks.Chunk()
	}
}

// Usage returns the last usage seen and whether any was found.
func (s *usageScanner) Usage() (tokenUsage, bool) {
	return s.last, s.found
}

// ensureStreamUsage injects stream_options.include_usage=true into a
// streaming request body when the caller didn't set include_usage,
// so vLLM emits the final usage chunk we log. Set-if-missing: a caller
// who explicitly sent include_usage (true or false) is respected.
//
// Client-visible effect when injected: the SSE stream gains one final
// chunk with empty choices and a usage object before [DONE]. This is
// standard OpenAI streaming-usage behavior and is ignored by
// OpenAI-compatible clients (they key off choices/[DONE]).
func ensureStreamUsage(body map[string]any) {
	so, ok := body["stream_options"].(map[string]any)
	if !ok {
		so = map[string]any{}
		body["stream_options"] = so
	}
	if _, exists := so["include_usage"]; !exists {
		so["include_usage"] = true
	}
}
