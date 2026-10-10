package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAuthConfigValidates(t *testing.T) {
	bad := []Config{
		{},
		{Mode: "magic"},
		{Mode: "static"},
		{Mode: "static", Keys: []StaticKey{{Key: "", User: "a"}}},
		{Mode: "static", Keys: []StaticKey{{Key: "k", User: "a"}, {Key: "k", User: "b"}}},
		{Mode: "external"},
	}
	for i, c := range bad {
		if c.Validate() == nil {
			t.Errorf("case %d (%+v) validated", i, c)
		}
	}
	good := []Config{
		{Mode: "none"},
		{Mode: "static", Keys: []StaticKey{{Key: "k", User: "a"}}},
		{Mode: "external", URL: "http://x"},
	}
	for i, c := range good {
		if err := c.Validate(); err != nil {
			t.Errorf("good case %d: %v", i, err)
		}
	}
}

func TestStaticAuth(t *testing.T) {
	a := New(Config{Mode: "static", Keys: []StaticKey{
		{Key: "alice-key", User: "alice"}, {Key: "batch-key", User: "batch", Priority: 10}}})
	ctx := context.Background()
	if r, _ := a.Authenticate(bearerReq(ctx, "batch-key")); r == nil || r.User != "batch" || r.Priority != 10 || r.Source != "static" {
		t.Fatalf("batch: %+v", r)
	}
	for _, tok := range []string{"", "alice-ke", "alice-key2", "nope"} {
		if r, _ := a.Authenticate(bearerReq(ctx, tok)); r != nil {
			t.Errorf("token %q accepted as %+v", tok, r)
		}
	}
}

func TestNoAuthAcceptsMissingToken(t *testing.T) {
	r, err := New(Config{Mode: "none"}).Authenticate(bearerReq(context.Background(), ""))
	if err != nil || r == nil || r.User != "anonymous" {
		t.Fatalf("%+v %v", r, err)
	}
}

func TestExternalAuthContract(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get("Authorization") {
		case "Bearer ok":
			w.Header().Set("X-Authenticated-User", "u")
			w.Header().Set("X-User-Priority", "5")
			w.WriteHeader(http.StatusNoContent)
		case "Bearer slow":
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
		case "Bearer anonymous":
			w.WriteHeader(http.StatusNoContent)
		case "Bearer badprio":
			w.Header().Set("X-Authenticated-User", "u")
			w.Header().Set("X-User-Priority", "low")
			w.WriteHeader(http.StatusNoContent)
		case "Bearer broken":
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.WriteHeader(http.StatusUnauthorized)
		}
	}))
	defer srv.Close()
	a := New(Config{Mode: "external", URL: srv.URL + "/"})
	ctx := context.Background()
	if r, err := a.Authenticate(bearerReq(ctx, "ok")); err != nil || r == nil || r.User != "u" || r.Priority != 5 || r.Source != "external" {
		t.Fatalf("ok: %+v %v", r, err)
	}
	if r, err := a.Authenticate(bearerReq(ctx, "bad")); r != nil || err != nil {
		t.Fatalf("bad: %+v %v", r, err)
	}
	_, err := a.Authenticate(bearerReq(ctx, "slow"))
	if rl, ok := err.(*RateLimitedError); !ok || rl.RetryAfter != 7 {
		t.Fatalf("slow: %v", err)
	}
	for _, tok := range []string{"anonymous", "badprio"} {
		if r, err := a.Authenticate(bearerReq(ctx, tok)); r != nil || err == nil {
			t.Errorf("%s: accepted as %+v (%v)", tok, r, err)
		}
	}
	if r, err := a.Authenticate(bearerReq(ctx, "broken")); r != nil || err == nil {
		t.Fatalf("broken: %+v %v", r, err)
	}
	if r, err := a.Authenticate(bearerReq(ctx, "")); r != nil || err != nil {
		t.Fatalf("missing token reached the backend: %+v %v", r, err)
	}
}

func bearerReq(ctx context.Context, token string) *http.Request {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil).WithContext(ctx)
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	return r
}

func TestTrustedHeader(t *testing.T) {
	const secret = "0123456789abcdef-secret"
	if err := (Config{Mode: "trusted_header", Secret: "short"}).Validate(); err == nil {
		t.Fatal("short secret accepted")
	}
	a := New(Config{Mode: "trusted_header", Secret: secret})
	req := func(h map[string]string) *http.Request {
		r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		for k, v := range h {
			r.Header.Set(k, v)
		}
		return r
	}
	if r, err := a.Authenticate(req(map[string]string{HeaderSecret: secret, HeaderUser: "alice", HeaderPriority: "10"})); err != nil || r == nil || r.User != "alice" || r.Priority != 10 {
		t.Fatalf("valid: %+v %v", r, err)
	}
	for name, h := range map[string]map[string]string{
		"no secret":    {HeaderUser: "alice"},
		"wrong secret": {HeaderSecret: secret + "x", HeaderUser: "alice"},
		"bearer only":  {"Authorization": "Bearer " + secret, HeaderUser: "alice"},
	} {
		if r, err := a.Authenticate(req(h)); r != nil || err != nil {
			t.Errorf("%s: %+v %v, want unknown caller", name, r, err)
		}
	}
	for name, h := range map[string]map[string]string{
		"no user":      {HeaderSecret: secret},
		"bad priority": {HeaderSecret: secret, HeaderUser: "alice", HeaderPriority: "high"},
	} {
		if r, err := a.Authenticate(req(h)); r != nil || err == nil {
			t.Errorf("%s: %+v %v, want an error (fails closed)", name, r, err)
		}
	}
}
