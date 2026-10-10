// The proxy handler: auth, alias rewrite, priority injection,
// stream-through forwarding to vLLM.
//
// Design notes:
//
//   - We only inspect/modify the body for endpoints that take a JSON
//     body with a `model` field (chat/completions, completions,
//     embeddings). Everything else under /v1/ is passed straight
//     through with the auth check applied. /v1/models is handled
//     specially so callers see *our* alias set, not raw upstream.
//
//   - Body rewrite is one-shot at request time. We read the request
//     body, mutate, and hand a new bytes.Reader to ReverseProxy. The
//     response stream is untouched: ReverseProxy handles SSE
//     correctly out of the box (flushes per write, propagates
//     cancellation via request Context).
//
//   - We log one structured event when the response completes. Status
//     code is captured via a ResponseWriter wrapper; latency is wall
//     time start→done. Token counts are not parsed in v1 (would
//     require tee'ing the SSE stream and matching the final usage
//     chunk); the architecture supports adding it later without
//     touching the streaming path.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

// Proxy carries the bits a request handler needs. Built once at
// startup; goroutine-safe.
//
// Callers are authenticated by the configured auth.Authenticator (auth.go).
type Proxy struct {
	cfg       *Config
	resolver  *ModelResolver
	upstream  *url.URL
	rp        *httputil.ReverseProxy
	logger    *EventLogger
	auth      auth.Authenticator
	admission *AdmissionQueue
	// tokenizeClient sizes requests for cache-aware admission; one-shot, no retries.
	tokenizeClient *http.Client
	tokenizeSem    chan struct{}
	// bodyBudget bounds total request-body bytes held at once. Bodies must be
	// read to rewrite the model alias, so size is bounded in aggregate rather than
	// per request; see bodybudget.go.
	bodyBudget *byteBudget
	health     *upstreamHealth

	// logRequestBodies, if true, emits the raw request body (pre-
	// rewrite) and the final body (post-rewrite) to stderr for every
	// body-rewrite path. Opt-in via the LOG_REQUEST_BODIES env var.
	// Used to diagnose client-side request shape issues (what is the
	// client actually sending, are sampling params or thinking flags
	// present, etc.). Off by default because bodies contain user
	// prompts and can be large.
	logRequestBodies bool
}

func NewProxy(cfg *Config, resolver *ModelResolver, logger *EventLogger) (*Proxy, error) {
	u, err := url.Parse(cfg.Upstream)
	if err != nil {
		return nil, err
	}
	health := newUpstreamHealth(cfg.Upstream, cfg.Retry, nil)
	p := &Proxy{
		cfg: cfg, resolver: resolver, upstream: u, logger: logger, auth: auth.New(cfg.Auth),
		admission:      NewAdmissionQueue(cfg.Admission),
		tokenizeClient: &http.Client{},
		tokenizeSem:    make(chan struct{}, 4),
		bodyBudget: newByteBudget(cfg.MaxInflightBodyBytes, cfg.BodyAmplification,
			cfg.MaxChunkedBodyBytes),
		health:           health,
		logRequestBodies: envFlag("LOG_REQUEST_BODIES"),
	}
	p.rp = &httputil.ReverseProxy{
		// Transport with retry-on-transient. The retryingTransport
		// re-issues the request on connection-refused / dial-timeout
		// (vLLM restart window) and on configured 5xx codes. POST
		// bodies survive the retry because rewriteBody() sets
		// req.GetBody before forwarding.
		Transport: newHealthAwareRetryingTransport(http.DefaultTransport, cfg.Retry, health),
		Director: func(r *http.Request) {
			r.URL.Scheme = u.Scheme
			r.URL.Host = u.Host
			r.Host = u.Host
			// vLLM exposes tokenization at /tokenize rather than under /v1.
			// Keep the public API consistently namespaced.
			if r.URL.Path == "/v1/tokenize" {
				r.URL.Path = "/tokenize"
			}
			// Strip our bearer header so upstream doesn't see it.
			// (vLLM ignores auth, but being clean costs nothing.)
			r.Header.Del("Authorization")
		},
		// Strip any CORS headers vLLM sets before the response reaches
		// the client. We set our own in ServeHTTP; letting upstream
		// headers through would produce duplicate values (e.g. "*, *").
		ModifyResponse: func(resp *http.Response) error {
			resp.Header.Del("Access-Control-Allow-Origin")
			resp.Header.Del("Access-Control-Allow-Methods")
			resp.Header.Del("Access-Control-Allow-Headers")
			resp.Header.Del("Access-Control-Allow-Credentials")
			resp.Header.Del("Access-Control-Max-Age")
			resp.Header.Del("Access-Control-Expose-Headers")
			return nil
		},
		// httputil.ReverseProxy already streams response bodies and
		// flushes per chunk for SSE. FlushInterval -1 forces immediate
		// flush on every Write — defensive against any future buffering
		// changes upstream.
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if errors.Is(err, ErrUpstreamUnavailable) {
				writeUpstreamUnavailable(w, health)
				return
			}
			log.Printf("level=warn msg=\"inference upstream proxy error\" err=%q path=%s", err.Error(), r.URL.Path)
			writeError(w, http.StatusBadGateway, "inference upstream request failed")
		},
	}
	return p, nil
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ev := Event{
		TS:     start.UTC().Format(time.RFC3339Nano),
		Method: r.Method,
		Path:   r.URL.Path,
	}

	// CORS: emit headers on every response so browser clients can reach
	// the API. OPTIONS preflight is answered before auth because the
	// browser never sends credentials on the preflight request.
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
	w.Header().Set("Access-Control-Max-Age", "86400")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	// Auth required for everything, including /v1/models (see below).
	who, authErr := p.checkAuth(r)
	// Credentials stop here: neither the gateway secret nor the identity headers reach
	// the engine.
	for _, h := range []string{auth.HeaderSecret, auth.HeaderUser, auth.HeaderPriority} {
		r.Header.Del(h)
	}
	if authErr != nil {
		// Only a rate limit from the auth backend arrives here; it is
		// returned verbatim.
		var kerr *auth.RateLimitedError
		if errors.As(authErr, &kerr) && kerr.RetryAfter > 0 {
			w.Header().Set("Retry-After", strconv.Itoa(kerr.RetryAfter))
		}
		status := http.StatusTooManyRequests
		if kerr != nil {
			status = kerr.Status
		}
		writeError(w, status, authErr.Error())
		ev.Status = status
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		return
	}
	if who == nil {
		writeError(w, http.StatusUnauthorized, "missing or invalid bearer token")
		ev.Status = http.StatusUnauthorized
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		return
	}
	ev.User = who.User
	ev.Auth = who.Source
	ev.Sub = who.Sub

	// /v1/models is answered locally by serveModels (synthetic alias list) and,
	// like every path here, requires an authenticated caller. Keyless liveness is
	// /readyz, which reports ready/not-ready without naming a model.
	// Cache-aware admission state: modelled occupancy and resident sessions (hashed).
	if r.Method == http.MethodGet && r.URL.Path == "/debug/residency" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(p.admission.Residency())
		ev.Status = http.StatusOK
		ev.LatencyMS = time.Since(start).Milliseconds()
		return
	}

	if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
		p.serveModels(w, r)
		ev.Status = http.StatusOK
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		return
	}

	// Known prolonged downtime fails before body rewriting, admission, and SSE
	// headers. This also avoids returning a stale alias-resolution error when
	// the proxy started while the upstream was already offline.
	if p.health != nil && !p.health.Accepting() {
		writeUpstreamUnavailable(w, p.health)
		ev.Status = http.StatusServiceUnavailable
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		return
	}

	// 2b. Reserve the memory this body will occupy before reading it. Admission
	// limits how many requests run at once; this limits how much of the host they
	// consume between them. The reservation is held until the upstream response is
	// done, because the body stays resident while it is being forwarded.
	//
	// Waiting here is deliberate and is not a new failure: a request that cannot
	// afford its own body waits, exactly as it would wait for an execution slot.
	// What it replaces is the old behavior of reading any size and then holding it
	// for a queue wait of up to thirty minutes.
	if p.shouldRewriteBody(r) {
		reserveStart := time.Now()
		release, err := p.bodyBudget.acquire(r.Context(), r.ContentLength)
		ev.BodyReserveMS = time.Since(reserveStart).Milliseconds()
		if err != nil {
			ev.ContentLength = r.ContentLength
			ev.Error = "in-flight body budget full: " + err.Error()
			writeError(w, http.StatusServiceUnavailable, ev.Error)
			ev.Status = http.StatusServiceUnavailable
			ev.LatencyMS = time.Since(start).Milliseconds()
			p.logger.Log(ev)
			return
		}
		defer release()
	}

	// 3. For body-bearing JSON endpoints, rewrite alias + inject priority.
	isStream := false
	if p.shouldRewriteBody(r) {
		var ok bool
		ok, isStream = p.rewriteBody(w, r, who, &ev)
		if !ok {
			ev.LatencyMS = time.Since(start).Milliseconds()
			p.logger.Log(ev)
			return // rewriteBody already wrote the error response
		}
	}

	// 4. Forward. Two paths:
	//   - Streaming chat/completions: we own the response so we can
	//     emit SSE keepalive comments while the retrying transport
	//     waits for a restarting upstream. See streaming.go.
	//   - Everything else (non-streaming, GET, etc.): ReverseProxy
	//     handles it; the retrying transport still covers connection
	//     errors but no keepalive is possible (can't send partial
	//     JSON, and non-streaming clients usually have long timeouts).
	if isStream {
		p.serveStreamingWithKeepalive(w, r, &ev)
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		return
	}

	if p.admission != nil && p.shouldAdmit(r) {
		release, _, wait, snap, ok := p.admission.Acquire(r.Context(), p.admissionRequest(r, &ev))
		ev.QueueMS = wait.Milliseconds()
		ev.RequestBytes = r.ContentLength
		ev.ActiveOnArrival = snap.Active
		ev.QueuedOnArrival = snap.Queued
		ev.CacheUsedOnArrival = snap.CacheUsed
		ev.applyAdmissionReport()
		if !ok {
			if r.Context().Err() != nil {
				return
			}
			status := http.StatusServiceUnavailable
			if wait == 0 {
				status = http.StatusTooManyRequests
			}
			writeError(w, status, "inference admission queue unavailable")
			ev.Status = status
			ev.LatencyMS = time.Since(start).Milliseconds()
			p.logger.Log(ev)
			return
		}
		defer release()
	}

	rw := &capturingWriter{ResponseWriter: w, status: 200, captureBody: shouldCaptureUsage(r.URL.Path)}
	p.rp.ServeHTTP(rw, r)
	ev.Status = rw.status
	if rw.captureBody && rw.status == http.StatusOK {
		if u, ok := parseUsageFromTail(rw.usageTail()); ok {
			ev.PromptTokens = u.Prompt
			ev.CompletionTokens = u.Completion
			ev.TotalTokens = u.Total
			ev.CachedPromptTokens = u.Cached
			// Set before the truncation flag: whether the engine told us about details at
			// all is a property of the response, not of how much of it we kept.
			ev.UsageDetails = u.Details
			ev.UsageTruncated = rw.truncated
		}
	}
	ev.LatencyMS = time.Since(start).Milliseconds()
	p.logger.Log(ev)
}

// checkAuth resolves the caller. Returns:
//
//	(result, nil)                 accepted
//	(nil, *auth.RateLimitedError)      rate limited; returned to the client verbatim
//	(nil, nil)                    unknown or missing token: 401
//
// A backend failure (unreachable, unexpected status) is logged and also
// answered 401, so an auth outage fails closed but stays visible.
func (p *Proxy) checkAuth(r *http.Request) (*auth.Result, error) {
	result, err := p.auth.Authenticate(r)
	if err != nil {
		var kerr *auth.RateLimitedError
		if errors.As(err, &kerr) {
			return nil, err
		}
		log.Printf("level=warn msg=\"auth backend failed\" err=%q", err.Error())
		return nil, nil
	}
	return result, nil
}

// shouldRewriteBody decides which routes carry a JSON body with a
// `model` field that we want to inspect. Keep the list narrow; future
// vLLM routes that don't take `model` (e.g. /v1/tokenize *does* take
// model, so it's in) should be added explicitly.
func (p *Proxy) shouldAdmit(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	switch r.URL.Path {
	case "/v1/chat/completions", "/v1/completions":
		return true
	}
	return false
}

func (p *Proxy) shouldRewriteBody(r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}
	switch r.URL.Path {
	case "/v1/chat/completions",
		"/v1/completions",
		"/v1/embeddings",
		"/tokenize",
		"/v1/tokenize":
		return true
	}
	return false
}

// rewriteBody reads the JSON request body, swaps the model name if
// it's an alias, and injects `priority` if non-zero. Returns (ok,
// isStreaming) where ok=false means the body was malformed and an
// error response has already been written. isStreaming reports
// whether the request body has `stream: true`, used by ServeHTTP
// to pick the right forwarding path (the streaming path emits SSE
// keepalives during upstream-retry windows).
//
// We decode into a generic map so we don't have to enumerate every
// field the OpenAI / vLLM body schemas support; we only touch
// `model` and `priority`, leaving the rest byte-equivalent.
func (p *Proxy) rewriteBody(w http.ResponseWriter, r *http.Request, who *auth.Result, ev *Event) (bool, bool) {
	// A declared length over the optional per-request cap is refused before the
	// body is read, so a rejection costs the client the upload it had not sent yet.
	// The cap is off by default (max_body_bytes: 0): image threads are legitimately
	// tens of megabytes, and the log showed a 34.7 MB thread rejected after the
	// client had already spent 32 seconds uploading it.
	raw, err := readBody(r, p.cfg.MaxBodyBytes, p.cfg.MaxChunkedBodyBytes)
	if err != nil && !errors.Is(err, errBodyOverBudget) {
		// Record how much of the body arrived against the declared length so
		// an aborted or timed-out upload is diagnosable from this log line.
		ev.RequestBytes = int64(len(raw))
		if r.ContentLength > 0 {
			ev.ContentLength = r.ContentLength
		}
		if errors.Is(err, errBodyOverBudget) && p.cfg.MaxBodyBytes > 0 {
			ev.Error = fmt.Sprintf("request body exceeds max_body_bytes (%d)", p.cfg.MaxBodyBytes)
			writeError(w, http.StatusRequestEntityTooLarge, ev.Error)
			ev.Status = http.StatusRequestEntityTooLarge
			return false, false
		}
		ev.Error = "read request body: " + err.Error()
		writeError(w, http.StatusBadRequest, ev.Error)
		ev.Status = http.StatusBadRequest
		return false, false
	}
	r.Body.Close()

	if p.logRequestBodies {
		logBodyDump(ev, "raw_in", raw)
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		ev.RequestBytes = int64(len(raw))
		ev.Error = "request body is not valid JSON"
		writeError(w, http.StatusBadRequest, ev.Error)
		ev.Status = http.StatusBadRequest
		return false, false
	}

	// vLLM Router currently rejects the OpenAI `developer` role before
	// forwarding the request, even though vLLM itself accepts it. Keep
	// this compatibility rewrite at the public API boundary so clients
	// do not need provider-specific workarounds. Remove it after router
	// PR https://github.com/vllm-project/router/pull/120 is merged and
	// the deployed router version has been validated with this role.
	if r.URL.Path == "/v1/chat/completions" {
		normalizeDeveloperRoles(body)
	}

	// Detect streaming. The JSON field `stream: true` selects SSE
	// output from vLLM; non-bool / absent means non-streaming.
	isStream, _ := body["stream"].(bool)

	// Resolve alias if the caller used one.
	requested, _ := body["model"].(string)
	ev.Alias = requested
	target, override, known := ResolveAlias(requested, p.cfg.Aliases, p.resolver)
	if known {
		if target == "" {
			writeError(w, http.StatusServiceUnavailable,
				"alias target unresolved; upstream model list not yet known")
			ev.Status = http.StatusServiceUnavailable
			return false, false
		}
		body["model"] = target
	}
	ev.Target, _ = body["model"].(string)

	// Fingerprint the conversation for passive request correlation. An explicit
	// X-Session-Key wins because it cannot collide across conversations; the
	// content hash is the fallback for clients that send nothing. This metadata
	// does not affect admission or vLLM cache behavior.
	if r.URL.Path == "/v1/chat/completions" {
		if id := r.Header.Get("X-Session-Key"); id != "" {
			ev.Session = sessionKeyFromID(ev.Target, who.Source+"/"+who.User+"/"+who.Sub, id)
		} else {
			ev.Session = sessionKey(ev.Target, who.Source+"/"+who.User+"/"+who.Sub, body)
		}
	}

	// Determine final priority (lower runs first). The caller's identity sets the base;
	// a caller may ask to run later (X-Priority header, else the body's priority field)
	// but never sooner than its identity allows, since ordering is strict and an
	// over-eager caller would starve everyone else. An alias priority override (the
	// caller picks the alias) replaces the request's value under the same floor.
	final := who.Priority
	if v, ok := callerPriority(r, body); ok {
		final = max(min(max(v, minCallerPriority), maxCallerPriority), who.Priority)
	}
	// Priority is the proxy's to set; the header is not forwarded upstream.
	r.Header.Del("X-Priority")
	if override != nil {
		final = max(*override, who.Priority)
	}
	ev.Priority = final
	if final != 0 {
		body["priority"] = final
	} else {
		delete(body, "priority")
	}

	// Inputs for cache-aware admission (cache.go). The body is kept for /tokenize, which
	// runs in the admission step, after a streaming client already gets keepalives.
	if p.shouldAdmit(r) {
		ev.maxTokens = maxTokensOf(body)
		ev.textBytes = textBytesOf(body)
		ev.admitBody = body
	}

	// Reasoning policy is chat-only: reasoning_effort is consumed while
	// rendering the chat template, so it has no meaning on raw /v1/completions
	// and the proxy has no basis to interpret it there. Both steps apply to
	// aliases and literal IDs alike, because neither overwrites a value the
	// caller deliberately chose and both reach the same template.
	if r.URL.Path == "/v1/chat/completions" {
		if mapping, exists := p.cfg.ReasoningEffortMappings[target]; exists {
			applyReasoningEffortMapping(body, mapping)
		}
		if effort, exists := p.cfg.ReasoningEffortDefaults[target]; exists {
			applyReasoningEffortDefault(body, effort)
		}
	}

	// Sampling policy is model-specific and applies to both generation routes.
	// Managed aliases apply it authoritatively; literal model IDs preserve
	// caller sampling values.
	//
	// This runs AFTER the reasoning steps on purpose: which preset applies
	// depends on whether the model will think, and that is only settled once
	// the effort has been translated and defaulted. Running it first would
	// pick the preset from the caller's raw body and miss the default we are
	// about to inject.
	if r.URL.Path == "/v1/chat/completions" || r.URL.Path == "/v1/completions" {
		if policy, exists := p.cfg.SamplingDefaults[target]; exists {
			// Raw /v1/completions renders no chat template, so no reasoning
			// scaffold is emitted whatever the body carries, and the
			// non-thinking row is the honest preset there.
			thinking := r.URL.Path == "/v1/chat/completions" && isThinkingEnabled(body)
			applySamplingDefaults(body, policy, known, thinking)
		}
	}

	// For streaming requests, ask vLLM to emit the final usage chunk so
	// we can log token counts (set-if-missing; see ensureStreamUsage).
	if isStream {
		ensureStreamUsage(body)
	}

	// Drop the original bytes before re-encoding. The decoded map owns its own
	// strings by now, so leaving raw live across the Marshal would hold both
	// copies at once at the peak of the request.
	raw = nil

	out, err := json.Marshal(body)
	if err != nil {
		// We just decoded this; re-encode failing would be a bug.
		writeError(w, http.StatusInternalServerError, "re-encode body: "+err.Error())
		ev.Status = http.StatusInternalServerError
		return false, false
	}
	if p.logRequestBodies {
		logBodyDump(ev, "rewritten_out", out)
	}
	r.Body = io.NopCloser(bytes.NewReader(out))
	r.ContentLength = int64(len(out))
	r.Header.Set("Content-Length", itoa(len(out)))
	// Body replay for the retrying transport: if the first upstream
	// attempt fails on a connection error (vLLM restart), the
	// transport asks GetBody for a fresh reader to retry with.
	// Without this, retries of POST endpoints would send an empty
	// body. Captures `out` by value via the closure.
	buf := out
	r.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(buf)), nil
	}
	return true, isStream
}

// normalizeDeveloperRoles maps OpenAI's newer developer role to system for
// compatibility with vLLM Router's currently strict chat-message parser.
// Malformed message entries are left untouched for the upstream to validate.
func normalizeDeveloperRoles(body map[string]any) {
	messages, ok := body["messages"].([]any)
	if !ok {
		return
	}
	for _, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if role, _ := message["role"].(string); role == "developer" {
			message["role"] = "system"
		}
	}
}

// applySamplingDefaults picks the preset for the requested thinking mode.
// Managed aliases use authoritative=true and overwrite configured keys.
// Literal IDs preserve caller keys, including meaningful explicit zeroes,
// and fill only keys that are absent.
func applySamplingDefaults(body map[string]any, sd ModelSamplingDefaults, authoritative, thinking bool) {
	defaults := sd.NoThinking
	if thinking {
		defaults = sd.Thinking
	}
	// A model that configures only one row has no mode-specific policy: that
	// row is simply its sampling policy, so use it either way. Without this,
	// a non-thinking model configured with just `no_thinking` would get no
	// policy at all, since thinking is the default for the models that do
	// distinguish. Both rows empty stays a no-op.
	if len(defaults) == 0 {
		if thinking {
			defaults = sd.NoThinking
		} else {
			defaults = sd.Thinking
		}
	}
	for k, v := range defaults {
		if authoritative {
			body[k] = v
			continue
		}
		if _, exists := body[k]; !exists {
			body[k] = v
		}
	}
}

// applyReasoningEffortMapping replaces an explicitly requested effort when
// the served model defines a translation for it. Missing and non-string
// values remain untouched for upstream validation.
func applyReasoningEffortMapping(body map[string]any, mapping map[string]string) {
	effort, ok := body["reasoning_effort"].(string)
	if !ok {
		return
	}
	if replacement, exists := mapping[effort]; exists {
		body["reasoning_effort"] = replacement
	}
}

// applyReasoningEffortDefault fills the effort for callers who omitted it, so
// an unspecified request means what the service chose rather than whatever the
// upstream chat template happens to default to. Any present value wins,
// including a non-string one, which is left for the upstream to reject.
//
// An explicit chat_template_kwargs.enable_thinking=false is also caller intent,
// expressed through the older per-template surface, so no default is added on
// top of it. vLLM only derives enable_thinking from reasoning_effort when the
// caller did not set it, so injecting here would be inert rather than harmful,
// but adding a second contradictory control would still be confusing and would
// depend on that precedence rule holding.
func applyReasoningEffortDefault(body map[string]any, effort string) {
	if _, exists := body["reasoning_effort"]; exists {
		return
	}
	if thinkingExplicitlyDisabled(body) {
		return
	}
	body["reasoning_effort"] = effort
}

// thinkingExplicitlyDisabled reports whether the caller turned thinking off
// through chat_template_kwargs. It differs from isThinkingEnabled, which cannot
// distinguish an absent flag from an explicit false.
func thinkingExplicitlyDisabled(body map[string]any) bool {
	ctk, ok := body["chat_template_kwargs"].(map[string]any)
	if !ok {
		return false
	}
	val, ok := ctk["enable_thinking"].(bool)
	return ok && !val
}

// isThinkingEnabled reports whether the request will produce reasoning.
//
// Thinking is assumed on unless turned off, as in chat templates like Qwen3's,
// where an absent reasoning_effort makes the template pick its top tier, so silence means thinking, not
// silence means off. Only two things turn it off, and both are explicit:
// reasoning_effort "none", or chat_template_kwargs.enable_thinking false.
//
// enable_thinking is checked first because it wins upstream too: vLLM only
// derives enable_thinking from reasoning_effort when the caller did not set
// it (chat_completion/protocol.py:587). That same precedence is why a
// reasoning_effort of "none" reaches the template at all instead of being
// rejected as an unsupported tier.
//
// Call this only for chat requests. Raw /v1/completions renders no template,
// so no reasoning is emitted regardless of what the body carries.
func isThinkingEnabled(body map[string]any) bool {
	if ctk, ok := body["chat_template_kwargs"].(map[string]any); ok {
		if val, ok := ctk["enable_thinking"].(bool); ok {
			return val
		}
	}
	if effort, ok := body["reasoning_effort"].(string); ok && effort == "none" {
		return false
	}
	return true
}

// serveModels returns a synthetic /v1/models response listing both
// the configured aliases and the literal upstream
// model id. The literal name was always passthrough-callable;
// including it in the list just makes it discoverable for callers who
// want to pin a specific model regardless of what an alias resolves to.
//
// owned_by distinguishes the two: aliases are "cavp",
// the literal upstream is "vllm". A client showing model lists can
// reasonably treat that as a grouping hint.
func (p *Proxy) serveModels(w http.ResponseWriter, r *http.Request) {
	type modelEntry struct {
		ID          string `json:"id"`
		Object      string `json:"object"`
		Created     int64  `json:"created"`
		OwnedBy     string `json:"owned_by"`
		Description string `json:"description,omitempty"`
		MaxModelLen int64  `json:"max_model_len,omitempty"`
	}
	now := time.Now().Unix()
	out := struct {
		Object string       `json:"object"`
		Data   []modelEntry `json:"data"`
	}{Object: "list"}
	maxModelLen := p.resolver.MaxModelLen()
	for name, alias := range p.cfg.Aliases {
		out.Data = append(out.Data, modelEntry{
			ID:          name,
			Object:      "model",
			Created:     now,
			OwnedBy:     "cavp",
			Description: alias.Description,
			MaxModelLen: maxModelLen,
		})
	}
	// Include the upstream's literal name when known. When the
	// resolver hasn't refreshed yet (or the upstream is down), we
	// just omit it; the aliases still work, and a missing entry is
	// less surprising than a placeholder.
	if up := p.resolver.Current(); up != "" {
		out.Data = append(out.Data, modelEntry{
			ID:          up,
			Object:      "model",
			Created:     now,
			OwnedBy:     "vllm",
			Description: "Literal upstream model id. Use it to pin a specific model version, bypassing alias indirection. This changes when the model is updated, at which point requests using this id will break until you update to the new id shown here.",
			MaxModelLen: maxModelLen,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// maxUsageCaptureBytes caps how much of a non-streaming response we
// buffer to parse its `usage` object. Chat/completions JSON is small;
// embeddings responses can be large (float arrays), so we stop
// capturing past the cap and skip usage parsing rather than buffer an
// unbounded body. The client still gets every byte (Write always
// forwards); only our side copy is bounded.
// maxUsageTailBytes is how much of the *end* of a response is kept to find the
// usage object. vLLM writes usage last in a JSON completion and in the final
// chunk of a stream, so the tail is the part worth keeping. The previous design
// kept the first 4 MiB and gave up past it, which silently dropped usage on long
// agentic responses: the biggest, most expensive turns were the ones with no
// token accounting, including the cached_tokens measurement.
const maxUsageTailBytes = 64 << 10

// capturingWriter remembers the response status for logging without
// breaking streaming. Implements http.Flusher so ReverseProxy can
// flush through it for SSE. When captureBody is set it also tees the
// response body (up to maxUsageCaptureBytes) so the caller can parse
// token usage after ServeHTTP returns.
type capturingWriter struct {
	http.ResponseWriter
	status      int
	captureBody bool
	tail        []byte
	truncated   bool
}

func (c *capturingWriter) WriteHeader(s int) {
	c.status = s
	c.ResponseWriter.WriteHeader(s)
}

func (c *capturingWriter) Write(b []byte) (int, error) {
	if c.captureBody {
		c.tail = append(c.tail, b...)
		// Trim at twice the target so the copy is amortised. Trimming on every
		// write would copy 64 KiB for each 50-byte SSE chunk, which is more CPU
		// than the response deserves.
		if len(c.tail) > 2*maxUsageTailBytes {
			c.tail = c.tail[len(c.tail)-maxUsageTailBytes:]
			c.truncated = true
		}
	}
	return c.ResponseWriter.Write(b)
}

// usageTail returns the kept tail, capped at maxUsageTailBytes.
func (c *capturingWriter) usageTail() []byte {
	if len(c.tail) > maxUsageTailBytes {
		return c.tail[len(c.tail)-maxUsageTailBytes:]
	}
	return c.tail
}

// usageCapturePaths are the non-streaming endpoints whose JSON bodies
// carry a `usage` object worth logging.
func shouldCaptureUsage(path string) bool {
	switch path {
	case "/v1/chat/completions", "/v1/completions", "/v1/embeddings":
		return true
	}
	return false
}

func (c *capturingWriter) Flush() {
	if f, ok := c.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "proxy_error"},
	})
}

func writeUpstreamUnavailable(w http.ResponseWriter, health *upstreamHealth) {
	retryAfter := 10
	if health != nil {
		retryAfter = health.RetryAfterSeconds()
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", strconv.Itoa(retryAfter))
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": "Inference is currently offline",
			"type":    "upstream_unavailable",
			"code":    "upstream_offline",
		},
	})
}

// envFlag returns true for env values "1", "true", "yes" (case-
// insensitive); false otherwise. Used by opt-in diagnostic knobs.
func envFlag(name string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	return v == "1" || v == "true" || v == "yes"
}

// logBodyDump prints a request-body dump line for diagnostic
// purposes. Truncates at maxBodyDumpBytes to keep log files
// manageable; vision payloads can be several MB and we never want
// those landing whole in stderr. The cap is generous because long-
// context agent traffic regularly exceeds 16 KB once tools + skills
// + a few prior turns are spliced in, and a truncated body is much
// less useful than the full picture for diagnosis. Vision payloads
// (base64 PNGs) will still hit the cap, which is intentional.
const maxBodyDumpBytes = 256 * 1024

func logBodyDump(ev *Event, label string, body []byte) {
	truncated := ""
	dump := body
	if len(dump) > maxBodyDumpBytes {
		dump = dump[:maxBodyDumpBytes]
		truncated = fmt.Sprintf(" (truncated from %d bytes)", len(body))
	}
	log.Printf("body_dump path=%s alias=%q target=%q user=%q sub=%q %s%s: %s",
		ev.Path, ev.Alias, ev.Target, ev.User, ev.Sub, label, truncated, dump)
}

// itoa avoids strconv for a value we know fits a small int.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// admissionRequest builds the admission inputs. It tokenizes the request first (exact size and the token IDs the KV mirror needs); on the streaming
// path this runs after SSE keepalives started. Observe reads ev at release time; both
// forwarding paths fill the usage fields before their deferred release runs.
func (p *Proxy) admissionRequest(r *http.Request, ev *Event) AdmissionRequest {
	report := &AdmissionReport{}
	ev.generated = p.admission.NewOutputCounter()
	ar := AdmissionRequest{
		Generated: ev.generated,
		Priority:  ev.Priority,
		Session:   ev.Session, MaxTokens: ev.maxTokens,
		Report: report,
		Observe: func() (CacheObservation, bool) {
			if ev.PromptTokens <= 0 || ev.Status != http.StatusOK {
				return CacheObservation{}, false
			}
			return CacheObservation{PromptTokens: ev.PromptTokens, CompletionTokens: ev.CompletionTokens}, true
		},
	}
	ev.report = report
	if body := ev.admitBody; body != nil {
		ev.admitBody = nil
		start := time.Now()
		tokens, err := p.tokenize(r.Context(), r.URL.Path, body, tokenizeTimeout)
		ev.TokenizeMS = time.Since(start).Milliseconds()
		if err == nil {
			ar.PromptTokens = len(tokens)
			ar.Blocks = p.admission.PromptKeys(tokens)
			ev.Tokenized = true
		} else {
			// Never charge less than the estimate: a 4xx may be a validation failure the
			// real request repeats, or a changed route that it does not.
			ar.PromptTokens = FallbackPrompt(ev.textBytes)
			ev.TokenizeError = err.Error()
		}
		ev.EstPromptTokens = ar.PromptTokens
	}
	return ar
}

// applyAdmissionReport copies what admission decided into the log event.
func (ev *Event) applyAdmissionReport() {
	if ev.report == nil {
		return
	}
	ev.ChargeTokens = ev.report.ChargeTokens
	ev.PredictedCachedTokens = ev.report.PredictedCached
	ev.PrefillCostS = math.Round(ev.report.PrefillCostS*10) / 10
	ev.DisplacedTokens = ev.report.DisplacedTokens
	ev.DisplaceCostS = math.Round(ev.report.DisplaceCostS*10) / 10
}

// callerPriority reads a caller-chosen priority: the X-Priority header, else a numeric
// priority field in the body.
func callerPriority(r *http.Request, body map[string]any) (int, bool) {
	if h := strings.TrimSpace(r.Header.Get("X-Priority")); h != "" {
		if v, err := strconv.Atoi(h); err == nil {
			return v, true
		}
	}
	if v, ok := body["priority"].(float64); ok && v == float64(int(v)) {
		return int(v), true
	}
	return 0, false
}

// maxTokensOf returns the request's output cap, or 0 when it has none.
func maxTokensOf(body map[string]any) int {
	for _, k := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := body[k].(float64); ok && v > 0 {
			return int(v)
		}
	}
	return 0
}

// textBytesOf is the size of a request body's text: inline media (data: URLs) is left
// out because its bytes say nothing about its token count.
func textBytesOf(v any) int64 {
	switch x := v.(type) {
	case string:
		if strings.HasPrefix(x, "data:") {
			return 0
		}
		return int64(len(x))
	case map[string]any:
		var n int64
		for k, e := range x {
			n += int64(len(k)) + textBytesOf(e)
		}
		return n
	case []any:
		var n int64
		for _, e := range x {
			n += textBytesOf(e)
		}
		return n
	}
	return 8
}

// tokenize asks the engine for the prompt's token IDs, rendering the chat template
// exactly as the real request will (same messages, tools and template kwargs; the
// engine applies its own defaults in both paths). status is the HTTP status when the
// engine answered.
func (p *Proxy) tokenize(ctx context.Context, path string, body map[string]any, timeout time.Duration) ([]int64, error) {
	req := map[string]any{"model": body["model"]}
	if path == "/v1/chat/completions" {
		for _, k := range []string{"messages", "tools", "chat_template_kwargs", "mm_processor_kwargs",
			"documents", "add_generation_prompt", "continue_final_message"} {
			if v, ok := body[k]; ok {
				req[k] = v
			}
		}
		if _, ok := req["add_generation_prompt"]; !ok {
			req["add_generation_prompt"] = true
		}
		// reasoning_effort is a chat completion field the template reads (it writes the
		// effort instruction at the start of the system prompt); /tokenize has no such
		// field, so it goes in as a template kwarg. vLLM lets the top-level field win over
		// a kwarg, and derives enable_thinking=false from "none" unless the caller set it.
		if eff, ok := body["reasoning_effort"]; ok {
			kw, _ := req["chat_template_kwargs"].(map[string]any)
			merged := map[string]any{}
			for k, v := range kw {
				merged[k] = v
			}
			merged["reasoning_effort"] = eff
			if _, set := merged["enable_thinking"]; !set && eff == "none" {
				merged["enable_thinking"] = false
			}
			req["chat_template_kwargs"] = merged
		}
	} else {
		req["prompt"] = body["prompt"]
	}
	buf, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	// The timeout covers waiting for a slot too.
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	// Bounded: tokenizing runs on the engine's API server, next to every stream.
	select {
	case p.tokenizeSem <- struct{}{}:
		defer func() { <-p.tokenizeSem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	u := *p.upstream
	u.Path = "/tokenize"
	hr, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	hr.Header.Set("Content-Type", "application/json")
	resp, err := p.tokenizeClient.Do(hr)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
		return nil, fmt.Errorf("tokenize HTTP %d", resp.StatusCode)
	}
	var out struct {
		Tokens []int64 `json:"tokens"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&out); err != nil {
		return nil, err
	}
	return out.Tokens, nil
}
