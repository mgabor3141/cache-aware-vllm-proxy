// Token usage, read from the engine's responses for the request log, the hit-ratio
// counters, and admission (a finished turn's prompt size becomes its session's hold).
//
//	Non-streaming: the response is one JSON object with a top-level "usage". The body
//	  is kept (up to maxUsageBodyBytes; the client gets every byte regardless) and
//	  decoded whole with encoding/json once it is complete.
//
//	Streaming: with stream_options.include_usage the engine sends a final chunk with
//	  empty choices and a usage object, just before "data: [DONE]". usageScanner reads
//	  the stream line by line as it passes. The proxy sets include_usage when the
//	  request does not (ensureStreamUsage); a caller that sets it false gets no usage
//	  line, and its turn leaves no hold.
package proxy

import (
	"bytes"
	"encoding/json"
)

// maxUsageBodyBytes bounds the copy of a non-streaming response kept to read its
// usage. A larger response is still forwarded in full; its usage is not logged.
const maxUsageBodyBytes = 16 << 20

// tokenUsage is the decoded usage of one response.
type tokenUsage struct {
	Prompt     int
	Completion int
	Total      int
	// Cached is how many prompt tokens the engine served from its prefix cache.
	Cached int
	// Details records whether the engine sent prompt_tokens_details at all, which
	// tells a real miss (Cached 0) from an engine that reports nothing.
	Details bool
}

// usageWire is the OpenAI/vLLM usage object.
type usageWire struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	PromptDetails    *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

func (u *usageWire) decoded() (tokenUsage, bool) {
	if u == nil || u.TotalTokens <= 0 {
		return tokenUsage{}, false
	}
	out := tokenUsage{Prompt: u.PromptTokens, Completion: u.CompletionTokens, Total: u.TotalTokens}
	if u.PromptDetails != nil {
		out.Cached, out.Details = u.PromptDetails.CachedTokens, true
	}
	return out, true
}

// parseResponseUsage reads usage from a complete non-streaming response body.
func parseResponseUsage(body []byte) (tokenUsage, bool) {
	var r struct {
		Usage *usageWire `json:"usage"`
	}
	if json.Unmarshal(body, &r) != nil {
		return tokenUsage{}, false
	}
	return r.Usage.decoded()
}

// maxSSELine bounds one pending SSE line. A longer one is dropped from scanning (not
// from the stream): no usage or [DONE] line is anywhere near this long.
const maxSSELine = 1 << 20

// usageScanner watches a passing SSE stream: it counts output (live output for
// admission: a chunk's usage when the stream reports it per chunk, else per choice chunk), keeps the last usage object and notes the [DONE] marker. Used by the one
// goroutine that copies the stream.
type usageScanner struct {
	chunks  *OutputCounter // when set, counts the output so far
	pending []byte
	skip    bool // pending is the tail of an over-long line
	usage   tokenUsage
	found   bool
	done    bool // saw "data: [DONE]"
}

func (s *usageScanner) Write(p []byte) {
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			if !s.skip {
				s.pending = append(s.pending, p...)
				if len(s.pending) > maxSSELine {
					s.pending, s.skip = s.pending[:0], true
				}
			}
			return
		}
		if !s.skip {
			s.pending = append(s.pending, p[:i]...)
			s.scanLine(s.pending)
		}
		s.pending, s.skip = s.pending[:0], false
		p = p[i+1:]
	}
}

func (s *usageScanner) scanLine(line []byte) {
	payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:"))
	if !ok {
		return
	}
	payload = bytes.TrimSpace(payload)
	if bytes.Equal(payload, []byte("[DONE]")) {
		s.done = true
		return
	}
	// Most chunks carry no usage (or "usage":null); decode only the ones that might.
	hasUsage := bytes.Contains(payload, []byte(`"usage"`)) && !bytes.Contains(payload, []byte(`"usage":null`))
	if s.chunks == nil && !hasUsage {
		return
	}
	var chunk struct {
		Choices []json.RawMessage `json:"choices"`
		Usage   *usageWire        `json:"usage"`
	}
	if json.Unmarshal(payload, &chunk) != nil {
		return
	}
	if u, ok := chunk.Usage.decoded(); ok {
		s.usage, s.found = u, true
		if s.chunks != nil && u.Completion > 0 {
			s.chunks.Usage(int64(u.Completion))
			return
		}
	}
	if s.chunks != nil && len(chunk.Choices) > 0 {
		s.chunks.Chunk()
	}
}

// ensureStreamUsage sets stream_options.include_usage when the request does not set
// it, so the engine sends the final usage chunk. A caller's explicit value is kept.
func ensureStreamUsage(body map[string]any) {
	so, ok := body["stream_options"].(map[string]any)
	if !ok {
		so = map[string]any{}
		body["stream_options"] = so
	}
	if _, set := so["include_usage"]; !set {
		so["include_usage"] = true
	}
}
