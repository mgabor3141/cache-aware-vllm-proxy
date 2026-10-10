package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

// These tests pin down the generation-policy contract the platform relies on.

func TestApplySamplingDefaults_PicksThinkingPreset(t *testing.T) {
	body := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": true},
	}
	sd := ModelSamplingDefaults{
		Thinking:   map[string]any{"temperature": 0.6, "top_p": 0.95},
		NoThinking: map[string]any{"temperature": 0.7, "top_p": 0.8},
	}
	applySamplingDefaults(body, sd, false, true)
	if got := body["temperature"]; got != 0.6 {
		t.Errorf("temperature=%v, want 0.6 (thinking preset)", got)
	}
	if got := body["top_p"]; got != 0.95 {
		t.Errorf("top_p=%v, want 0.95 (thinking preset)", got)
	}
}

func TestApplySamplingDefaults_PicksNoThinkingPreset_WhenAbsent(t *testing.T) {
	body := map[string]any{}
	sd := ModelSamplingDefaults{
		Thinking:   map[string]any{"temperature": 0.6},
		NoThinking: map[string]any{"temperature": 0.7, "top_p": 0.8},
	}
	applySamplingDefaults(body, sd, false, false)
	if got := body["temperature"]; got != 0.7 {
		t.Errorf("temperature=%v, want 0.7 (no_thinking preset, thinking absent)", got)
	}
}

func TestApplySamplingDefaults_PicksNoThinkingPreset_WhenFalse(t *testing.T) {
	body := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}
	sd := ModelSamplingDefaults{
		Thinking:   map[string]any{"temperature": 0.6},
		NoThinking: map[string]any{"temperature": 0.7},
	}
	applySamplingDefaults(body, sd, false, false)
	if got := body["temperature"]; got != 0.7 {
		t.Errorf("temperature=%v, want 0.7 (no_thinking preset)", got)
	}
}

func TestApplySamplingDefaults_DoesNotOverwriteCaller(t *testing.T) {
	// Critical contract: anything the caller sent stays. Even explicit
	// zero (greedy decoding is a legitimate request).
	body := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": true},
		"temperature":          0.0,
		"top_p":                0.42,
	}
	sd := ModelSamplingDefaults{
		Thinking: map[string]any{
			"temperature": 0.6,
			"top_p":       0.95,
			"top_k":       20,
		},
	}
	applySamplingDefaults(body, sd, false, true)
	if got := body["temperature"]; got != 0.0 {
		t.Errorf("temperature=%v, want 0.0 (caller's value); defaults overwrote", got)
	}
	if got := body["top_p"]; got != 0.42 {
		t.Errorf("top_p=%v, want 0.42 (caller's value); defaults overwrote", got)
	}
	// top_k was unset; default should land
	if got := body["top_k"]; got != 20 {
		t.Errorf("top_k=%v, want 20 (default should fill missing key)", got)
	}
}

func TestApplySamplingDefaults_EmptyConfigIsNoOp(t *testing.T) {
	// A proxy with no sampling_defaults configured must not mutate the
	// body. Existing deployments + tests that don't set the new block
	// should keep their pre-change behavior.
	body := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": true},
		"model":                "default",
	}
	original := map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": true},
		"model":                "default",
	}
	applySamplingDefaults(body, ModelSamplingDefaults{}, false, true)
	if len(body) != len(original) {
		t.Errorf("body grew under empty defaults: %v", body)
	}
}

func TestApplyReasoningEffortMapping(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
		want any
	}{
		{"translated effort", map[string]any{"reasoning_effort": "high"}, "xhigh"},
		{"translated minimal", map[string]any{"reasoning_effort": "minimal"}, "low"},
		{"untranslated effort", map[string]any{"reasoning_effort": "low"}, "low"},
		{"adaptive medium survives", map[string]any{"reasoning_effort": "medium"}, "medium"},
		{"thinking off survives", map[string]any{"reasoning_effort": "none"}, "none"},
		{"missing effort", map[string]any{}, nil},
		{"malformed effort", map[string]any{"reasoning_effort": 1}, 1},
	}
	mapping := map[string]string{"minimal": "low", "high": "xhigh"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			applyReasoningEffortMapping(tt.body, mapping)
			if got := tt.body["reasoning_effort"]; got != tt.want {
				t.Errorf("reasoning_effort=%v, want %v", got, tt.want)
			}
		})
	}
}

func TestApplyReasoningEffortDefault(t *testing.T) {
	tests := []struct {
		name string
		body map[string]any
		want any
	}{
		{"absent gets service default", map[string]any{}, "medium"},
		{"explicit value wins", map[string]any{"reasoning_effort": "low"}, "low"},
		{"explicit none is not treated as absent", map[string]any{"reasoning_effort": "none"}, "none"},
		{"malformed value is left for upstream", map[string]any{"reasoning_effort": 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			applyReasoningEffortDefault(tt.body, "medium")
			if got := tt.body["reasoning_effort"]; got != tt.want {
				t.Errorf("reasoning_effort=%v, want %v", got, tt.want)
			}
		})
	}
}

// Translation runs before defaulting, so a caller-sent value is translated and
// a caller-omitted one is filled. The default is expected to be a value the
// upstream already accepts, so it is not translated afterwards.
func TestReasoningTranslationThenDefaultOrdering(t *testing.T) {
	mapping := map[string]string{"minimal": "low", "high": "xhigh"}

	sent := map[string]any{"reasoning_effort": "high"}
	applyReasoningEffortMapping(sent, mapping)
	applyReasoningEffortDefault(sent, "medium")
	if got := sent["reasoning_effort"]; got != "xhigh" {
		t.Errorf("sent effort=%v, want xhigh", got)
	}

	omitted := map[string]any{}
	applyReasoningEffortMapping(omitted, mapping)
	applyReasoningEffortDefault(omitted, "medium")
	if got := omitted["reasoning_effort"]; got != "medium" {
		t.Errorf("omitted effort=%v, want medium", got)
	}
}

// Thinking is the DEFAULT for this model family, so the interesting cases are
// the two explicit ways to turn it off and the precedence between them. An
// earlier version of this helper answered "off" for everything except an
// explicit enable_thinking=true, which inverted the model's actual behaviour
// and silently selected the non-thinking sampling row for ordinary requests.
func TestIsThinkingEnabled(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
		want bool
	}{
		// Silence means thinking: the template picks its top tier.
		{"empty body", map[string]any{}, true},
		{"empty chat_template_kwargs", map[string]any{"chat_template_kwargs": map[string]any{}}, true},
		{"enable_thinking absent", map[string]any{"chat_template_kwargs": map[string]any{"other": 1}}, true},

		// The two explicit ways off.
		{"enable_thinking false", map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}, false},
		{"reasoning_effort none", map[string]any{"reasoning_effort": "none"}, false},

		// Ordinary tiers all think.
		{"reasoning_effort low", map[string]any{"reasoning_effort": "low"}, true},
		{"reasoning_effort medium", map[string]any{"reasoning_effort": "medium"}, true},
		{"reasoning_effort xhigh", map[string]any{"reasoning_effort": "xhigh"}, true},

		// enable_thinking wins over reasoning_effort, matching vLLM's own
		// precedence (chat_completion/protocol.py:587).
		{"enable_thinking true beats none", map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": true},
			"reasoning_effort":     "none",
		}, true},
		{"enable_thinking false beats medium", map[string]any{
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
			"reasoning_effort":     "medium",
		}, false},

		// Malformed values are not caller intent to disable; leave them to
		// the upstream and keep the default.
		{"enable_thinking string (malformed)", map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": "true"}}, true},
		{"chat_template_kwargs wrong type", map[string]any{"chat_template_kwargs": "x"}, true},
		{"reasoning_effort wrong type", map[string]any{"reasoning_effort": 1}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isThinkingEnabled(c.body); got != c.want {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

// A model that configures only one row has no mode-specific policy. Selecting
// the empty row and applying nothing would silently drop sampling policy for
// every non-thinking model, which is how this nearly regressed: thinking is
// the default, so the thinking row is what gets selected by default.
func TestApplySamplingDefaults_SingleRowAppliesToBothModes(t *testing.T) {
	only := ModelSamplingDefaults{NoThinking: map[string]any{"temperature": 0.7}}
	for _, thinking := range []bool{true, false} {
		body := map[string]any{}
		applySamplingDefaults(body, only, false, thinking)
		if got := body["temperature"]; got != 0.7 {
			t.Errorf("thinking=%v: temperature=%v, want 0.7", thinking, got)
		}
	}

	onlyThinking := ModelSamplingDefaults{Thinking: map[string]any{"temperature": 1.0}}
	for _, thinking := range []bool{true, false} {
		body := map[string]any{}
		applySamplingDefaults(body, onlyThinking, false, thinking)
		if got := body["temperature"]; got != 1.0 {
			t.Errorf("thinking=%v: temperature=%v, want 1.0", thinking, got)
		}
	}
}

// End-to-end through rewriteBody: the preset must follow the effort the proxy
// itself injects, which is only correct because sampling runs after the
// reasoning steps. And /v1/completions renders no template, so it takes the
// non-thinking row whatever the body carries.
func TestRewriteBody_SamplingFollowsReasoningEffort(t *testing.T) {
	newProxy := func() *Proxy {
		return &Proxy{cfg: &Config{
			Aliases: map[string]AliasEntry{"managed": {Target: "qwen"}},
			SamplingDefaults: SamplingDefaults{"qwen": {
				Thinking:   map[string]any{"temperature": 1.0},
				NoThinking: map[string]any{"temperature": 0.7},
			}},
			ReasoningEffortDefaults: map[string]string{"qwen": "medium"},
		}}
	}
	cases := []struct {
		name, path, body string
		wantTemp         any
	}{
		{"omitted effort defaults to medium, so thinking row",
			"/v1/chat/completions", `{"model":"qwen"}`, 1.0},
		{"explicit none takes the non-thinking row",
			"/v1/chat/completions", `{"model":"qwen","reasoning_effort":"none"}`, 0.7},
		{"enable_thinking false takes the non-thinking row",
			"/v1/chat/completions", `{"model":"qwen","chat_template_kwargs":{"enable_thinking":false}}`, 0.7},
		{"ordinary tier takes the thinking row",
			"/v1/chat/completions", `{"model":"qwen","reasoning_effort":"low"}`, 1.0},
		{"raw completions takes the non-thinking row",
			"/v1/completions", `{"model":"qwen"}`, 0.7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			if ok, _ := newProxy().rewriteBody(rec, req, &auth.Result{}, &Event{}); !ok {
				t.Fatalf("rewrite failed: %s", rec.Body.String())
			}
			var got map[string]any
			if err := json.NewDecoder(req.Body).Decode(&got); err != nil {
				t.Fatal(err)
			}
			if got["temperature"] != tc.wantTemp {
				t.Errorf("temperature=%v, want %v; body=%v", got["temperature"], tc.wantTemp, got)
			}
		})
	}
}

// Raw /v1/completions must not acquire a reasoning_effort: it has no chat
// template to consume one.
func TestRewriteBody_CompletionsGetsNoReasoningEffort(t *testing.T) {
	p := &Proxy{cfg: &Config{
		Aliases:                 map[string]AliasEntry{"managed": {Target: "qwen"}},
		ReasoningEffortDefaults: map[string]string{"qwen": "medium"},
	}}
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(`{"model":"qwen"}`))
	rec := httptest.NewRecorder()
	if ok, _ := p.rewriteBody(rec, req, &auth.Result{}, &Event{}); !ok {
		t.Fatalf("rewrite failed: %s", rec.Body.String())
	}
	var got map[string]any
	if err := json.NewDecoder(req.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if _, exists := got["reasoning_effort"]; exists {
		t.Errorf("completions acquired reasoning_effort: %v", got)
	}
}
