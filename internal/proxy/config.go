// Config types and loader. One YAML file, read at startup; changes need a restart.
package proxy

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
	"gopkg.in/yaml.v3"
)

// Config is the entire on-disk configuration. Anything that varies
// between deployments lives here; everything else is code.
type Config struct {
	// Listen is the address the HTTP server binds to (plain HTTP; put TLS
	// in front if it is exposed).
	Listen string `yaml:"listen"`

	// Upstream is the vLLM base URL; /v1/... is appended per request.
	// One engine: cache-aware admission models a single KV pool.
	Upstream string `yaml:"upstream"`

	// ModelsRefreshInterval controls how often we re-query the upstream
	// /v1/models endpoint to resolve aliases marked as "<auto>". The
	// served model only changes on a redeploy, so a generous interval is
	// fine; short enough that swaps take effect without intervention.
	ModelsRefreshInterval time.Duration `yaml:"models_refresh_interval"`

	// MaxBodyBytes is an optional per-request body cap. The default, zero, means
	// no per-request limit.
	//
	// There used to be a hard 32 MiB cap here. It produced a 400 on a multi-image
	// thread that was 34.7 MB, after the client had spent 32 seconds uploading it,
	// and the size that triggered it was ordinary usage rather than abuse. A fixed
	// per-request number is the wrong shape for multimodal traffic: any value is
	// either a cliff for someone's real thread or high enough to be no protection.
	// The aggregate budget below is the protection, and it does not depend on
	// guessing how big a legitimate request can be.
	MaxBodyBytes int64 `yaml:"max_body_bytes"`

	// MaxInflightBodyBytes bounds the request-body bytes the proxy holds at once,
	// across all requests. This is what makes an unlimited per-request size safe:
	// admission allows max_active requests running and any number queued, and a body read
	// before admission stays resident for the whole queue wait, which can be thirty
	// minutes. Without an aggregate bound, body size and queue depth multiply.
	MaxInflightBodyBytes int64 `yaml:"max_inflight_body_bytes"`

	// BodyAmplification charges each body against the budget at this multiple of
	// its declared length, because the buffered path holds more than one copy: the
	// raw bytes, the decoded map, and the re-encoded bytes. Measured by
	// TestDecodeAmplification rather than estimated.
	BodyAmplification int64 `yaml:"body_amplification"`

	// MaxChunkedBodyBytes caps and prices a body that arrives with no declared
	// Content-Length. A length is what makes a reservation possible, so an unnamed
	// body cannot be planned for and gets this fixed treatment instead: charged to
	// the budget at this size and refused past it. Every client in front of this
	// proxy sends Content-Length; this branch exists so an unknown-length upload is
	// the conservative case rather than the one that slips past accounting.
	MaxChunkedBodyBytes int64 `yaml:"max_chunked_body_bytes"`

	// Aliases maps user-facing model names to backend targets, with
	// optional priority overrides. The literal model name vLLM serves
	// is always allowed through without an alias entry.
	Aliases map[string]AliasEntry `yaml:"aliases"`

	// Retry controls how the proxy reacts to transient upstream
	// failures (vLLM restarting, brief 5xx during graceful shutdown,
	// etc.). See retry.go for full semantics. Optional in YAML; the
	// defaults applied by RetryConfig.applyDefaults cover the longest
	// observed cold restart while bounding the shared outage episode.
	Retry RetryConfig `yaml:"retry"`

	// SamplingDefaults maps literal model IDs to their model-specific
	// thinking and non-thinking policies. Managed aliases apply the
	// selected policy authoritatively; literal IDs use set-if-missing
	// semantics as an expert escape hatch. Unknown models get no policy.
	SamplingDefaults SamplingDefaults `yaml:"sampling_defaults"`

	// ReasoningEffortMappings translates effort names the served model does not
	// accept into ones it does. It applies to aliases and literal IDs because
	// the upstream model has the same chat template on both paths, and an
	// unsupported value fails identically either way. Translation only rewrites
	// a value the caller actually sent.
	ReasoningEffortMappings map[string]map[string]string `yaml:"reasoning_effort_mappings"`

	// ReasoningEffortDefaults chooses the effort for callers who omit the field
	// entirely, so the service rather than the model's chat template decides what
	// an unspecified request means. Set-if-missing on both the alias and literal
	// paths: absence carries no caller intent to preserve, so there is nothing to
	// distinguish. Applied after translation, and the configured value is
	// expected to be one the upstream already accepts.
	ReasoningEffortDefaults map[string]string `yaml:"reasoning_effort_defaults"`

	// Admission bounds active LLM request lifetimes and queues excess
	// requests in the proxy (admission.go, cache.go).
	Admission AdmissionConfig `yaml:"admission"`

	// Auth selects how callers are authenticated (auth.go).
	Auth auth.Config `yaml:"auth"`
}

// SamplingDefaults is keyed by the literal upstream model ID so a model swap
// cannot accidentally inherit the previous model's sampling recipe.
type SamplingDefaults map[string]ModelSamplingDefaults

type ModelSamplingDefaults struct {
	Thinking   map[string]any `yaml:"thinking"`
	NoThinking map[string]any `yaml:"no_thinking"`
}

// AliasEntry describes one virtual model name. Target may be the literal
// upstream model name, or the sentinel "<auto>" which means "whatever
// the upstream is currently serving" — useful for the `default` alias
// so it survives model swaps without proxy config changes.
//
// PriorityOverride, when non-nil, sets the priority of every request
// through the alias, never more urgent than the caller's own (a `batch`
// alias makes work yield to interactive traffic). *int so "unset" and
// "explicit zero" are distinct.
//
// Thinking on or off (`chat_template_kwargs`) stays a client concern.
// Per-model sampling defaults (SamplingDefaults) are authoritative
// through an alias and fill-in-only with the literal model name, since
// vLLM's generic defaults suit some models badly.
type AliasEntry struct {
	Target           string `yaml:"target"`
	PriorityOverride *int   `yaml:"priority_override,omitempty"`

	// Description is surfaced verbatim on /v1/models as an extra field
	// (OpenAI's schema doesn't define one; harmless to add), so the docs
	// about each alias live next to the alias itself.
	Description string `yaml:"description,omitempty"`
}

// AutoTarget is the sentinel string for dynamic resolution.
const AutoTarget = "<auto>"

// LoadConfig reads and validates a config file. Validation surfaces
// problems at startup rather than at first request: duplicate keys,
// empty user fields, malformed durations all fail fast.
//
// The file is passed through os.ExpandEnv before YAML parsing so
// secrets can live in the environment rather than in the file. Anything
// like ${MY_KEY} becomes the matching variable's value; a missing one
// expands to empty (which the validator then catches as an empty key).
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}
	expanded := os.ExpandEnv(string(data))
	c := &Config{}
	// Unknown keys are errors: a misspelt or retired setting would otherwise be
	// ignored silently and the proxy would run on its default.
	dec := yaml.NewDecoder(strings.NewReader(expanded))
	dec.KnownFields(true)
	if err := dec.Decode(c); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	return c, c.validate()
}

func (c *Config) validate() error {
	if c.Listen == "" {
		c.Listen = ":4001"
	}
	if c.Upstream == "" {
		return fmt.Errorf("upstream is required")
	}
	if err := c.Auth.Validate(); err != nil {
		return err
	}
	c.Upstream = strings.TrimRight(c.Upstream, "/")
	if c.ModelsRefreshInterval <= 0 {
		c.ModelsRefreshInterval = 60 * time.Second
	}

	if c.MaxBodyBytes < 0 {
		return fmt.Errorf("max_body_bytes must be non-negative (0 means unlimited)")
	}
	if c.MaxInflightBodyBytes <= 0 {
		c.MaxInflightBodyBytes = 3 << 30
	}
	if c.BodyAmplification <= 0 {
		c.BodyAmplification = defaultBodyAmplification
	}
	if c.MaxChunkedBodyBytes < 0 {
		return fmt.Errorf("max_chunked_body_bytes must be non-negative")
	}
	if c.MaxChunkedBodyBytes == 0 {
		c.MaxChunkedBodyBytes = defaultChunkedBodyBytes
	}
	for name, a := range c.Aliases {
		if a.Target == "" {
			return fmt.Errorf("aliases[%q]: target is empty", name)
		}
	}
	c.Retry.applyDefaults()
	if c.Retry.RecoveryProbeInterval >= c.Retry.RecoveryTimeout {
		return fmt.Errorf("retry.recovery_probe_interval %s must be shorter than retry.recovery_timeout %s",
			c.Retry.RecoveryProbeInterval, c.Retry.RecoveryTimeout)
	}
	c.Admission.applyDefaults()
	if err := c.Admission.validate(); err != nil {
		return err
	}
	return nil
}
