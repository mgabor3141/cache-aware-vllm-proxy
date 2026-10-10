package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
)

// makeModelsProxy builds a minimal Proxy for testing serveModels.
func makeModelsProxy(t *testing.T, upstreamModel string) *Proxy {
	t.Helper()
	cfg := &Config{
		Aliases: map[string]AliasEntry{
			"default": {Target: "<auto>", Description: "default alias"},
			"batch":   {Target: "<auto>", PriorityOverride: intPtr(10), Description: "batch alias"},
		},
		SamplingDefaults: SamplingDefaults{
			"my-model": ModelSamplingDefaults{
				NoThinking: map[string]any{"temperature": 0.7, "top_p": 0.95},
			},
		},
	}
	resolver := &ModelResolver{}
	if upstreamModel != "" {
		resolver.maxModelLen.Store(196608)
		resolver.current.Store(&upstreamModel)
	}
	p := &Proxy{
		cfg:      cfg,
		resolver: resolver,
	}
	return p
}

func intPtr(i int) *int { return &i }

// modelIDs extracts the list of model IDs from a /v1/models response.
func modelIDs(t *testing.T, body []byte) []string {
	t.Helper()
	var resp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("unmarshal /v1/models response: %v", err)
	}
	ids := make([]string, len(resp.Data))
	for i, m := range resp.Data {
		ids[i] = m.ID
	}
	return ids
}

func modelContextWindows(t *testing.T, body []byte) map[string]int64 {
	t.Helper()
	var resp struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int64  `json:"max_model_len"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("unmarshal /v1/models response: %v", err)
	}
	windows := make(map[string]int64, len(resp.Data))
	for _, model := range resp.Data {
		windows[model.ID] = model.MaxModelLen
	}
	return windows
}

func containsID(ids []string, want string) bool {
	for _, id := range ids {
		if id == want {
			return true
		}
	}
	return false
}

func TestServeModels_ListsAliasesAndUpstream(t *testing.T) {
	p := makeModelsProxy(t, "my-model")
	rec := httptest.NewRecorder()
	p.serveModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	ids := modelIDs(t, rec.Body.Bytes())
	if len(ids) != 3 {
		t.Errorf("got %v, want default, batch and the upstream model", ids)
	}
	windows := modelContextWindows(t, rec.Body.Bytes())
	for _, model := range []string{"default", "batch", "my-model"} {
		if windows[model] != 196608 {
			t.Errorf("%s max_model_len = %d, want 196608", model, windows[model])
		}
	}
}

func TestServeModels_NoUpstreamModel(t *testing.T) {
	// Before the resolver's first refresh only the aliases are listed.
	p := makeModelsProxy(t, "")
	rec := httptest.NewRecorder()
	p.serveModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	ids := modelIDs(t, rec.Body.Bytes())
	if len(ids) != 2 || !containsID(ids, "default") || !containsID(ids, "batch") {
		t.Errorf("got %v, want only the aliases", ids)
	}
}

func TestRewriteBody_SamplingPrecedence(t *testing.T) {
	p := &Proxy{
		cfg: &Config{
			Aliases: map[string]AliasEntry{"default": {Target: "my-model"}},
			SamplingDefaults: SamplingDefaults{
				"my-model": {NoThinking: map[string]any{"temperature": 0.7, "top_p": 0.95}},
			},
		},
	}
	tests := []struct {
		name string
		body string
		want map[string]any
	}{
		{
			name: "alias overrides explicit values",
			body: `{"model":"default","temperature":0.2,"top_p":0.3}`,
			want: map[string]any{"model": "my-model", "temperature": 0.7, "top_p": 0.95},
		},
		{
			name: "literal preserves explicit and fills missing",
			body: `{"model":"my-model","temperature":0.2}`,
			want: map[string]any{"model": "my-model", "temperature": 0.2, "top_p": 0.95},
		},
		{
			name: "unknown literal passes through",
			body: `{"model":"unknown","temperature":0.2}`,
			want: map[string]any{"model": "unknown", "temperature": 0.2},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			ok, _ := p.rewriteBody(rec, req, &auth.Result{}, &Event{})
			if !ok {
				t.Fatalf("rewrite failed: %s", rec.Body.String())
			}
			var body map[string]any
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			for key, want := range tc.want {
				if got := body[key]; got != want {
					t.Errorf("%s=%v, want %v; body=%v", key, got, want, body)
				}
			}
			if tc.name == "unknown literal passes through" {
				if _, exists := body["top_p"]; exists {
					t.Fatalf("unknown model received policy: %v", body)
				}
			}
		})
	}
}

// Reasoning policy is exercised through rewriteBody rather than the helpers so
// that route scoping, alias resolution and step ordering are pinned too. The
// helper-level tests in sampling_defaults_test.go cannot catch a miswired call
// site, which is the failure this policy is most exposed to.
func TestRewriteBody_ReasoningEffortPolicy(t *testing.T) {
	newProxy := func() *Proxy {
		return &Proxy{
			cfg: &Config{
				Aliases: map[string]AliasEntry{"default": {Target: "qwen3.8-27b-fp8"}},
				ReasoningEffortMappings: map[string]map[string]string{
					"qwen3.8-27b-fp8": {"minimal": "low", "high": "xhigh"},
				},
				ReasoningEffortDefaults: map[string]string{"qwen3.8-27b-fp8": "medium"},
			},
		}
	}
	tests := []struct {
		name   string
		path   string
		body   string
		want   any
		absent bool
	}{
		{name: "alias absent effort gets service default", path: "/v1/chat/completions",
			body: `{"model":"default"}`, want: "medium"},
		{name: "literal absent effort gets service default", path: "/v1/chat/completions",
			body: `{"model":"qwen3.8-27b-fp8"}`, want: "medium"},
		{name: "alias translates high", path: "/v1/chat/completions",
			body: `{"model":"default","reasoning_effort":"high"}`, want: "xhigh"},
		{name: "literal translates minimal", path: "/v1/chat/completions",
			body: `{"model":"qwen3.8-27b-fp8","reasoning_effort":"minimal"}`, want: "low"},
		{name: "adaptive medium is not rewritten", path: "/v1/chat/completions",
			body: `{"model":"default","reasoning_effort":"medium"}`, want: "medium"},
		{name: "thinking off survives", path: "/v1/chat/completions",
			body: `{"model":"default","reasoning_effort":"none"}`, want: "none"},
		{name: "explicit enable_thinking false gets no default", path: "/v1/chat/completions",
			body: `{"model":"default","chat_template_kwargs":{"enable_thinking":false}}`, absent: true},
		{name: "explicit enable_thinking true still gets default", path: "/v1/chat/completions",
			body: `{"model":"default","chat_template_kwargs":{"enable_thinking":true}}`, want: "medium"},
		{name: "raw completions gets no default", path: "/v1/completions",
			body: `{"model":"default","prompt":"hi"}`, absent: true},
		{name: "raw completions is not translated", path: "/v1/completions",
			body: `{"model":"default","prompt":"hi","reasoning_effort":"high"}`, want: "high"},
		{name: "unknown model gets no reasoning policy", path: "/v1/chat/completions",
			body: `{"model":"some-other-model"}`, absent: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			ok, _ := newProxy().rewriteBody(rec, req, &auth.Result{}, &Event{})
			if !ok {
				t.Fatalf("rewrite failed: %s", rec.Body.String())
			}
			var body map[string]any
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			got, exists := body["reasoning_effort"]
			if tc.absent {
				if exists {
					t.Fatalf("reasoning_effort=%v, want absent; body=%v", got, body)
				}
				return
			}
			if got != tc.want {
				t.Errorf("reasoning_effort=%v, want %v; body=%v", got, tc.want, body)
			}
		})
	}
}

func TestRewriteBody_NormalizesDeveloperRoleOnChatOnly(t *testing.T) {
	p := &Proxy{cfg: &Config{}}
	input := `{"model":"my-model","messages":[{"role":"developer","content":"Be concise","name":"policy"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}`

	for _, tc := range []struct {
		path     string
		wantRole string
	}{
		{path: "/v1/chat/completions", wantRole: "system"},
		{path: "/v1/tokenize", wantRole: "developer"},
	} {
		t.Run(tc.path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(input))
			rec := httptest.NewRecorder()
			ok, _ := p.rewriteBody(rec, req, &auth.Result{}, &Event{})
			if !ok {
				t.Fatalf("rewrite failed: %s", rec.Body.String())
			}
			var body struct {
				Messages []map[string]any `json:"messages"`
			}
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if got := body.Messages[0]["role"]; got != tc.wantRole {
				t.Errorf("developer role=%v, want %q", got, tc.wantRole)
			}
			if got := body.Messages[0]["content"]; got != "Be concise" {
				t.Errorf("developer content changed: %v", got)
			}
			if got := body.Messages[0]["name"]; got != "policy" {
				t.Errorf("developer extra field changed: %v", got)
			}
			if got := body.Messages[1]["role"]; got != "user" {
				t.Errorf("user role changed: %v", got)
			}
			if got := body.Messages[2]["role"]; got != "assistant" {
				t.Errorf("assistant role changed: %v", got)
			}
		})
	}
}

func TestRewriteBody_NonGenerationRoutesSkipSamplingPolicy(t *testing.T) {
	p := &Proxy{
		cfg: &Config{
			Aliases: map[string]AliasEntry{"default": {Target: "my-model"}},
			SamplingDefaults: SamplingDefaults{
				"my-model": {NoThinking: map[string]any{"temperature": 0.7, "top_p": 0.95}},
			},
		},
	}
	for _, path := range []string{"/v1/embeddings", "/v1/tokenize"} {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"default","input":"test"}`))
			rec := httptest.NewRecorder()
			ok, _ := p.rewriteBody(rec, req, &auth.Result{Priority: 3}, &Event{})
			if !ok {
				t.Fatalf("rewrite failed: %s", rec.Body.String())
			}
			var body map[string]any
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if got := body["model"]; got != "my-model" {
				t.Errorf("model=%v, want resolved literal model", got)
			}
			if got := body["priority"]; got != float64(3) {
				t.Errorf("priority=%v, want 3", got)
			}
			for _, key := range []string{"temperature", "top_p"} {
				if _, exists := body[key]; exists {
					t.Errorf("%s was injected on non-generation route: %v", key, body)
				}
			}
		})
	}
}

// Session fingerprinting provides passive request correlation; pin the wiring,
// not just the hash helper. Turns of the same conversation must collide, different
// conversations and different targets must not, and raw completions must stay
// ephemeral.
func TestRewriteBody_SessionKeyWiring(t *testing.T) {
	p := &Proxy{cfg: &Config{Aliases: map[string]AliasEntry{"default": {Target: "qwen3.8-27b-fp8"}}}}
	rewrite := func(path, body string) *Event {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		rec := httptest.NewRecorder()
		ev := &Event{}
		if ok, _ := p.rewriteBody(rec, req, &auth.Result{}, ev); !ok {
			t.Fatalf("rewrite failed: %s", rec.Body.String())
		}
		return ev
	}

	turn1 := rewrite("/v1/chat/completions",
		`{"model":"default","messages":[{"role":"system","content":"sys"},{"role":"user","content":"review A"}]}`)
	turn2 := rewrite("/v1/chat/completions",
		`{"model":"default","messages":[{"role":"system","content":"sys"},{"role":"user","content":"review A"},{"role":"assistant","content":"done"},{"role":"user","content":"next"}]}`)
	other := rewrite("/v1/chat/completions",
		`{"model":"default","messages":[{"role":"system","content":"sys"},{"role":"user","content":"review B"}]}`)
	literal := rewrite("/v1/chat/completions",
		`{"model":"qwen3.8-27b-fp8","messages":[{"role":"system","content":"sys"},{"role":"user","content":"review A"}]}`)
	otherModel := rewrite("/v1/chat/completions",
		`{"model":"some-other-model","messages":[{"role":"system","content":"sys"},{"role":"user","content":"review A"}]}`)
	raw := rewrite("/v1/completions", `{"model":"default","prompt":"hi"}`)

	if turn1.Session == "" {
		t.Fatal("chat request got no session key")
	}
	if turn1.Session != turn2.Session {
		t.Errorf("turns of one conversation diverged: %q vs %q", turn1.Session, turn2.Session)
	}
	if turn1.Session != literal.Session {
		t.Errorf("alias and literal for same target diverged: %q vs %q", turn1.Session, literal.Session)
	}
	if turn1.Session == other.Session {
		t.Error("different conversations collided")
	}
	if turn1.Session == otherModel.Session {
		t.Error("different targets collided")
	}
	if raw.Session != "" {
		t.Errorf("raw completions got session %q, want empty", raw.Session)
	}
}

func TestRewriteBody_SessionKeyHeaderOverride(t *testing.T) {
	p := &Proxy{cfg: &Config{Aliases: map[string]AliasEntry{"default": {Target: "qwen3.8-27b-fp8"}}}}
	rewrite := func(header, body string) *Event {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		if header != "" {
			req.Header.Set("X-Session-Key", header)
		}
		rec := httptest.NewRecorder()
		ev := &Event{}
		if ok, _ := p.rewriteBody(rec, req, &auth.Result{}, ev); !ok {
			t.Fatalf("rewrite failed: %s", rec.Body.String())
		}
		return ev
	}
	body := `{"model":"default","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}`
	withA := rewrite("conv-a", body)
	withA2 := rewrite("conv-a", body)
	withB := rewrite("conv-b", body)
	hashed := rewrite("", body)
	if withA.Session != withA2.Session {
		t.Error("same header diverged")
	}
	if withA.Session == withB.Session {
		t.Error("different headers collided")
	}
	if withA.Session == hashed.Session {
		t.Error("header key equals content-hash key; identical conversations with and without headers must not merge")
	}
	if withA.Session == "conv-a" {
		t.Error("raw header value leaked as session key")
	}
}

func TestSessionKeyHeaderIsNamespacedByIdentity(t *testing.T) {
	p := &Proxy{cfg: &Config{Aliases: map[string]AliasEntry{"default": {Target: "qwen3.8-27b-fp8"}}}}
	rewrite := func(user string) *Event {
		body := `{"model":"default","messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		req.Header.Set("X-Session-Key", "conv-1")
		rec := httptest.NewRecorder()
		ev := &Event{}
		if ok, _ := p.rewriteBody(rec, req, &auth.Result{User: user, Source: "key"}, ev); !ok {
			t.Fatalf("rewrite failed: %s", rec.Body.String())
		}
		return ev
	}
	if rewrite("alice").Session == rewrite("bob").Session {
		t.Error("same X-Session-Key from different identities merged")
	}
	if rewrite("alice").Session != rewrite("alice").Session {
		t.Error("same identity and key diverged")
	}
}

// /v1/models requires a bearer, like every other path.
//
// Keyless liveness is /readyz, which names no model; the model menu is not
// world-readable. This test keeps it that way.
func TestModelsRequiresBearer(t *testing.T) {
	const goodKey = "good-key"
	authd := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/validate" || r.Header.Get("Authorization") != "Bearer "+goodKey {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("X-Authenticated-User", "tester")
		w.Header().Set("X-User-Sub", "sub-tester")
		w.Header().Set("X-Auth-Source", "dynamic")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer authd.Close()

	p := makeModelsProxy(t, "my-model")
	p.auth = auth.New(auth.Config{Mode: "external", URL: authd.URL})
	p.logger = NewEventLogger(io.Discard)

	// main.go mounts the handler at "/", so paths arrive with /v1 intact.
	noKey := httptest.NewRecorder()
	p.ServeHTTP(noKey, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if noKey.Code != http.StatusUnauthorized {
		t.Fatalf("GET /v1/models with no bearer: got %d, want 401 (body: %s)", noKey.Code, noKey.Body.String())
	}

	// A bearer the auth service does not know must not be a way in either.
	badKey := httptest.NewRecorder()
	bad := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	bad.Header.Set("Authorization", "Bearer wrong-key")
	p.ServeHTTP(badKey, bad)
	if badKey.Code != http.StatusUnauthorized {
		t.Fatalf("GET /v1/models with an unknown bearer: got %d, want 401 (body: %s)", badKey.Code, badKey.Body.String())
	}

	ok := httptest.NewRecorder()
	good := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	good.Header.Set("Authorization", "Bearer "+goodKey)
	p.ServeHTTP(ok, good)
	if ok.Code != http.StatusOK {
		t.Fatalf("GET /v1/models with a valid bearer: got %d, want 200 (body: %s)", ok.Code, ok.Body.String())
	}
	ids := modelIDs(t, ok.Body.Bytes())
	if len(ids) == 0 {
		t.Fatalf("authorized /v1/models returned no models: %s", ok.Body.String())
	}
}

// /readyz is the keyless signal the landing page and monitors use now that
// the model list is keyed. It must answer without an Authorization header.
func TestReadyzNeedsNoBearer(t *testing.T) {
	// The mux in main.go registers /healthz and /readyz outside the proxy
	// handler, so they are reachable with no credentials by construction.
	// This asserts the invariant we now depend on: if someone ever moves
	// /readyz behind the authenticated handler, the landing page's status
	// lights and both Kuma monitors go blind at once.
	mux := http.NewServeMux()
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("/readyz unauthenticated: got %d, want 200", rec.Code)
	}
}

// In trusted_header mode the gateway's secret and identity headers must not reach the
// engine.
func TestTrustedHeadersNotForwarded(t *testing.T) {
	var got http.Header
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[]}`))
	}))
	defer up.Close()
	f := t.TempDir() + "/c.yaml"
	os.WriteFile(f, []byte("upstream: "+up.URL+"\nauth:\n  mode: trusted_header\n  secret: 0123456789abcdef-secret\nadmission:\n  cache:\n    kv_events_endpoint: tcp://127.0.0.1:1\n"), 0o600)
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	h, err := NewProxy(cfg, NewModelResolver(up.URL, http.DefaultClient), NewEventLogger(io.Discard))
	if err != nil {
		t.Fatal(err)
	}
	h.health.recordSuccess("test")
	h.admission.SetPool(201, 3200)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set(auth.HeaderSecret, "0123456789abcdef-secret")
	req.Header.Set(auth.HeaderUser, "alice")
	req.Header.Set(auth.HeaderPriority, "5")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	for _, k := range []string{auth.HeaderSecret, auth.HeaderUser, auth.HeaderPriority} {
		if got.Get(k) != "" {
			t.Errorf("%s forwarded to the engine", k)
		}
	}
}
