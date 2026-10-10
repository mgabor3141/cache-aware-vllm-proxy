// The request path: routing, authentication, body parsing, waiting for the engine and
// for admission, and forwarding.
//
// Routes (anything else is 404):
//
//	POST /v1/chat/completions, /v1/completions   admitted (cache.go), then forwarded
//	POST /v1/tokenize, /tokenize                 forwarded to the engine's /tokenize
//	GET  /v1/models                              forwarded
//	GET  /healthz, /readyz, /metrics             keyless; aggregate state only
//
// Every /v1 and /tokenize request is authenticated and logged as one JSON line
// (logging.go). Only a few request headers reach the engine (upstreamHeaders); the
// caller's credentials never do.
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
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

const (
	// keepaliveInterval: a waiting or silent stream gets an SSE comment at least this
	// often, so clients and intermediaries with idle timeouts keep the connection.
	keepaliveInterval = 10 * time.Second
	// writeStallTimeout bounds one write to a client. A client that stops reading
	// (without disconnecting) would otherwise hold its admission permit forever.
	writeStallTimeout = 5 * time.Minute
)

// Proxy is the HTTP handler. Built once at startup; safe for concurrent use.
type Proxy struct {
	cfg       *Config
	upstream  *url.URL
	client    *http.Client // to the engine; no timeout, the request's context rules
	logger    *EventLogger
	auth      auth.Authenticator
	admission *AdmissionQueue
	ready     *readiness
	metrics   requestMetrics

	tokenizeClient *http.Client
	tokenizeSem    chan struct{}

	keepalive  time.Duration
	writeStall time.Duration

	closing   chan struct{} // closed when the server shuts down
	closeOnce sync.Once
}

func NewProxy(cfg *Config, logger *EventLogger) (*Proxy, error) {
	u, err := url.Parse(cfg.Upstream)
	if err != nil {
		return nil, err
	}
	authn, err := auth.New(cfg.Auth)
	if err != nil {
		return nil, err
	}
	return &Proxy{
		cfg: cfg, upstream: u, logger: logger, auth: authn,
		client:         &http.Client{},
		admission:      NewAdmissionQueue(cfg.Admission),
		ready:          newReadiness(cfg.Upstream, cfg.Readiness.Interval),
		tokenizeClient: &http.Client{},
		tokenizeSem:    make(chan struct{}, 4),
		keepalive:      keepaliveInterval,
		writeStall:     writeStallTimeout,
		closing:        make(chan struct{}),
	}, nil
}

// Close makes every request still waiting for the engine or for admission give up with
// 503; requests already running continue. Called when the server shuts down.
func (p *Proxy) Close() {
	p.closeOnce.Do(func() { close(p.closing) })
}

// apiRoute names the API route of a request, or "" when there is none.
func apiRoute(r *http.Request) string {
	switch r.Method + " " + r.URL.Path {
	case "POST /v1/chat/completions":
		return "chat"
	case "POST /v1/completions":
		return "completions"
	case "POST /v1/tokenize", "POST /tokenize":
		return "tokenize"
	case "GET /v1/models":
		return "models"
	}
	return ""
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		switch r.URL.Path {
		case "/healthz":
			io.WriteString(w, "ok\n")
			return
		case "/readyz":
			if !p.ready.Ready() {
				http.Error(w, "engine not ready", http.StatusServiceUnavailable)
				return
			}
			io.WriteString(w, "ok\n")
			return
		case "/metrics":
			p.serveMetrics(w)
			return
		}
	}

	start := time.Now()
	route := apiRoute(r)
	ev := &Event{TS: start.UTC().Format(time.RFC3339Nano), Method: r.Method, Path: r.URL.Path}
	defer func() {
		ev.LatencyMS = time.Since(start).Milliseconds()
		p.logger.Log(ev)
		p.metrics.observe(route, ev)
	}()

	if route == "" {
		ev.Status = http.StatusNotFound
		writeError(w, ev.Status, "not found")
		return
	}
	who, err := p.auth.Authenticate(r)
	if err != nil {
		log.Printf("level=warn msg=\"refused a malformed credential\" err=%q", err.Error())
	}
	if who == nil {
		ev.Status = http.StatusUnauthorized
		writeError(w, ev.Status, "missing or invalid credentials")
		return
	}
	ev.User, ev.Auth = who.User, who.Source

	var body []byte
	if r.Method == http.MethodPost {
		var ok bool
		if body, ok = p.readBody(w, r, ev); !ok {
			return
		}
	}

	// Bound every write: see writeStallTimeout. The deadline is cleared at the end, as
	// the connection may serve further requests.
	rc := http.NewResponseController(w)
	defer rc.SetWriteDeadline(time.Time{})
	w = &stallGuard{ResponseWriter: w, rc: rc, d: p.writeStall}

	switch route {
	case "models":
		p.forward(w, r, ev, "/v1/models", nil)
	case "tokenize":
		p.forward(w, r, ev, "/tokenize", body)
	default:
		p.serveInference(w, r, ev, who, body)
	}
}

// readBody reads the request body, at most max_body_bytes of it. On failure it has
// answered the client: 413 for a body over the limit (refused before reading when the
// declared length already says so), 400 for one that could not be read.
func (p *Proxy) readBody(w http.ResponseWriter, r *http.Request, ev *Event) ([]byte, bool) {
	limit := p.cfg.MaxBodyBytes
	tooLarge := func() ([]byte, bool) {
		ev.ContentLength = r.ContentLength
		ev.Status, ev.Error = http.StatusRequestEntityTooLarge, fmt.Sprintf("request body exceeds max_body_bytes (%d)", limit)
		writeError(w, ev.Status, ev.Error)
		return nil, false
	}
	if r.ContentLength > limit {
		return tooLarge()
	}
	var buf bytes.Buffer
	// Sized for the declared length, up to a point: memory follows bytes that arrived,
	// not bytes that were promised.
	if r.ContentLength > 0 {
		buf.Grow(int(min(r.ContentLength, 4<<20)) + bytes.MinRead)
	}
	_, err := buf.ReadFrom(http.MaxBytesReader(w, r.Body, limit))
	ev.RequestBytes = int64(buf.Len())
	if err != nil {
		if errors.As(err, new(*http.MaxBytesError)) {
			return tooLarge()
		}
		ev.ContentLength = r.ContentLength
		ev.Status, ev.Error = http.StatusBadRequest, "read request body: "+err.Error()
		writeError(w, ev.Status, ev.Error)
		return nil, false
	}
	return buf.Bytes(), true
}

// serveInference handles an admitted route: parse and amend the body, then forward it
// as a stream or as one response.
func (p *Proxy) serveInference(w http.ResponseWriter, r *http.Request, ev *Event, who *auth.Result, raw []byte) {
	var body map[string]any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber() // numbers pass through exactly as sent
	if err := dec.Decode(&body); err != nil || body == nil {
		ev.Status, ev.Error = http.StatusBadRequest, "request body is not a JSON object"
		writeError(w, ev.Status, ev.Error)
		return
	}
	raw = nil

	ev.Model, _ = body["model"].(string)
	// Only a JSON true streams; anything else is a single response.
	ev.Stream = body["stream"] == true
	if r.URL.Path == "/v1/chat/completions" {
		ev.Session = sessionKey(ev.Model, who.Source+"/"+who.User, body)
	}

	// Priority (lower runs first): the caller's identity sets the most urgent it may be;
	// the body's priority field may only make it run later, since ordering is strict and
	// an over-eager caller would starve everyone else.
	ev.Priority = who.Priority
	if v, ok := intField(body, "priority"); ok {
		ev.Priority = max(min(max(v, minCallerPriority), maxCallerPriority), who.Priority)
	}
	if ev.Priority != 0 {
		body["priority"] = ev.Priority
	} else {
		delete(body, "priority")
	}
	// The final usage chunk of a stream is how the proxy learns a turn's size, which
	// becomes its session's hold.
	if ev.Stream {
		ensureStreamUsage(body)
	}
	ev.maxTokens = maxTokensOf(body)
	ev.textBytes = textBytesOf(body)
	ev.body = body

	out, err := json.Marshal(body)
	if err != nil {
		ev.Status, ev.Error = http.StatusBadRequest, "re-encode body: "+err.Error()
		writeError(w, ev.Status, ev.Error)
		return
	}
	if ev.Stream {
		p.serveStream(w, r, ev, out)
		return
	}

	release, err := p.admit(r.Context(), r, ev)
	if err != nil {
		waitOutcome(ev, err)
		if ev.Status == http.StatusServiceUnavailable {
			writeError(w, ev.Status, "the proxy is shutting down; retry")
		}
		return
	}
	defer release() // runs after the usage below is recorded: admission reads it
	keep := &bodyKeeper{limit: maxUsageBodyBytes}
	p.forwardTo(w, r, ev, r.URL.Path, out, keep)
	if ev.Status == http.StatusOK && !keep.over {
		if u, ok := parseResponseUsage(keep.buf); ok {
			ev.setUsage(u)
		}
	}
}

var (
	errClientGone   = errors.New("client went away while waiting")
	errShuttingDown = errors.New("shutting down")
)

// admit waits for the engine to be ready and for an admission permit, however long
// that takes. It fails only when the client leaves or the server shuts down.
func (p *Proxy) admit(ctx context.Context, r *http.Request, ev *Event) (func(), error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		select {
		case <-p.closing:
			cancel()
		case <-ctx.Done():
		}
	}()
	gone := func() error {
		select {
		case <-p.closing:
			return errShuttingDown
		default:
			return errClientGone
		}
	}

	start := time.Now()
	if p.ready.Wait(ctx) != nil {
		ev.ReadyWaitMS = time.Since(start).Milliseconds()
		return nil, gone()
	}
	ev.ReadyWaitMS = time.Since(start).Milliseconds()

	release, _, wait, snap, ok := p.admission.Acquire(ctx, p.admissionRequest(ctx, r, ev))
	ev.QueueMS = wait.Milliseconds()
	ev.ActiveOnArrival, ev.QueuedOnArrival, ev.CacheUsedOnArrival = snap.Active, snap.Queued, snap.CacheUsed
	ev.applyAdmissionReport()
	if !ok {
		return nil, gone()
	}
	// The engine may have gone away while this request queued. Wait for it again,
	// holding the permit: nothing runs meanwhile, and the request keeps its turn.
	if !p.ready.Ready() {
		start := time.Now()
		err := p.ready.Wait(ctx)
		ev.ReadyWaitMS += time.Since(start).Milliseconds()
		if err != nil {
			release()
			return nil, gone()
		}
	}
	return release, nil
}

// waitOutcome records a request that left while waiting: 499 when its client went away
// (nothing to answer), 503 when the server is shutting down.
func waitOutcome(ev *Event, err error) {
	ev.Status, ev.Error = statusClientGone, err.Error()
	if errors.Is(err, errShuttingDown) {
		ev.Status = http.StatusServiceUnavailable
	}
}

// upstreamHeaders are the request headers forwarded to the engine. Everything else
// (credentials, identity headers, hop-by-hop and proxy headers) stays here.
var upstreamHeaders = []string{"Content-Type", "Accept", "User-Agent", "X-Request-Id"}

// hopHeaders are response headers that describe one connection and are not forwarded.
var hopHeaders = []string{"Connection", "Keep-Alive", "Proxy-Connection", "Proxy-Authenticate",
	"Te", "Trailer", "Transfer-Encoding", "Upgrade"}

// upstreamRequest builds the request to the engine at path.
func (p *Proxy) upstreamRequest(ctx context.Context, r *http.Request, path string, body []byte) (*http.Request, error) {
	u := *p.upstream
	u.Path = strings.TrimRight(u.Path, "/") + path
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, r.Method, u.String(), rd)
	if err != nil {
		return nil, err
	}
	for _, h := range upstreamHeaders {
		if v := r.Header.Values(h); len(v) > 0 {
			req.Header[h] = v
		}
	}
	return req, nil
}

func (p *Proxy) forward(w http.ResponseWriter, r *http.Request, ev *Event, path string, body []byte) {
	p.forwardTo(w, r, ev, path, body, nil)
}

// forwardTo sends the request to the engine and relays the response. keep, when set,
// also receives the response body. A response cut short aborts the client's connection,
// so the client cannot mistake it for a complete one.
func (p *Proxy) forwardTo(w http.ResponseWriter, r *http.Request, ev *Event, path string, body []byte, keep *bodyKeeper) {
	req, err := p.upstreamRequest(r.Context(), r, path, body)
	if err != nil {
		ev.Status, ev.Error = http.StatusInternalServerError, err.Error()
		writeError(w, ev.Status, "internal error")
		return
	}
	resp, err := p.client.Do(req)
	if err != nil {
		if r.Context().Err() != nil {
			ev.Status, ev.Error = statusClientGone, "client went away"
			return
		}
		ev.Status, ev.Error = http.StatusBadGateway, "engine request failed: "+err.Error()
		writeError(w, ev.Status, "inference upstream request failed")
		return
	}
	defer resp.Body.Close()
	ev.Status = resp.StatusCode
	h := w.Header()
	for k, v := range resp.Header {
		h[k] = v
	}
	for _, k := range hopHeaders {
		h.Del(k)
	}
	w.WriteHeader(resp.StatusCode)
	buf := make([]byte, 32<<10)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			keep.Write(buf[:n])
			if _, werr := w.Write(buf[:n]); werr != nil {
				ev.Error = "client went away: " + werr.Error()
				return
			}
		}
		if rerr == io.EOF {
			return
		}
		if rerr != nil {
			ev.Error = "engine response cut: " + rerr.Error()
			if r.Context().Err() != nil {
				ev.Error = "client went away"
				return
			}
			panic(http.ErrAbortHandler)
		}
	}
}

// bodyKeeper keeps a copy of a response body up to limit; over it, it keeps nothing.
type bodyKeeper struct {
	limit int
	buf   []byte
	over  bool
}

func (k *bodyKeeper) Write(b []byte) {
	if k == nil || k.over {
		return
	}
	if len(k.buf)+len(b) > k.limit {
		k.buf, k.over = nil, true
		return
	}
	k.buf = append(k.buf, b...)
}

// stallGuard gives each write to the client writeStallTimeout to complete.
type stallGuard struct {
	http.ResponseWriter
	rc *http.ResponseController
	d  time.Duration
}

func (s *stallGuard) Write(b []byte) (int, error) {
	_ = s.rc.SetWriteDeadline(time.Now().Add(s.d))
	return s.ResponseWriter.Write(b)
}

func (s *stallGuard) Flush() {
	_ = s.rc.SetWriteDeadline(time.Now().Add(s.d))
	_ = s.rc.Flush()
}

func (s *stallGuard) FlushError() error {
	_ = s.rc.SetWriteDeadline(time.Now().Add(s.d))
	return s.rc.Flush()
}

func (s *stallGuard) Unwrap() http.ResponseWriter { return s.ResponseWriter }

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "proxy_error", "code": status},
	})
}

// admissionRequest builds the admission inputs. It tokenizes the prompt first (exact
// size, and the token IDs the KV mirror needs); on the streaming path keepalives are
// already flowing. Observe reads ev when the permit is released, after the forwarding
// path recorded the status and usage.
func (p *Proxy) admissionRequest(ctx context.Context, r *http.Request, ev *Event) AdmissionRequest {
	ev.report = &AdmissionReport{}
	ev.generated = p.admission.NewOutputCounter()
	ar := AdmissionRequest{
		Priority: ev.Priority, Session: ev.Session, MaxTokens: ev.maxTokens,
		Generated: ev.generated, Report: ev.report,
		Observe: func() (CacheObservation, bool) {
			if ev.PromptTokens <= 0 || ev.Status != http.StatusOK {
				return CacheObservation{}, false
			}
			return CacheObservation{PromptTokens: ev.PromptTokens, CompletionTokens: ev.CompletionTokens}, true
		},
	}
	body := ev.body
	ev.body = nil
	if body == nil {
		return ar
	}
	start := time.Now()
	tokens, err := p.tokenize(ctx, r.URL.Path, body, tokenizeTimeout)
	ev.TokenizeMS = time.Since(start).Milliseconds()
	if err == nil {
		ar.PromptTokens = len(tokens)
		ar.Tokens = tokens
		ev.Tokenized = true
	} else {
		// Never charge less than the estimate: a 4xx may be a validation failure the
		// real request repeats, or a changed route that it does not.
		ar.PromptTokens = FallbackPrompt(ev.textBytes)
		ev.TokenizeError = err.Error()
	}
	ev.EstPromptTokens = ar.PromptTokens
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

// intField reads an integer field of the body.
func intField(body map[string]any, key string) (int, bool) {
	n, ok := body[key].(json.Number)
	if !ok {
		return 0, false
	}
	v, err := n.Int64()
	if err != nil || v != int64(int(v)) {
		return 0, false
	}
	return int(v), true
}

// maxTokensOf returns the request's output cap, or 0 when it has none.
func maxTokensOf(body map[string]any) int {
	for _, k := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := intField(body, k); ok && v > 0 {
			return v
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
// engine applies its own defaults in both paths).
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
	u.Path = strings.TrimRight(u.Path, "/") + "/tokenize"
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
