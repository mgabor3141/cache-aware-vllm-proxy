package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func mustNew(t *testing.T, c Config) Authenticator {
	t.Helper()
	a, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestAuthConfigValidates(t *testing.T) {
	bad := []Config{
		{},
		{Mode: "magic"},
		{Mode: "external"},
		{Mode: "static"},
		{Mode: "static", Keys: []StaticKey{{Key: "", User: "a"}}},
		{Mode: "static", Keys: []StaticKey{{Key: "k", User: "a"}, {Key: "k", User: "b"}}},
	}
	for i, c := range bad {
		if c.Validate() == nil {
			t.Errorf("case %d (%+v) validated", i, c)
		}
	}
	good := []Config{
		{Mode: "none"},
		{Mode: "static", Keys: []StaticKey{{Key: "k", User: "a"}}},
		{Mode: "trusted_header", Secret: "0123456789abcdef"},
	}
	for i, c := range good {
		if err := c.Validate(); err != nil {
			t.Errorf("good case %d: %v", i, err)
		}
	}
}

// An unknown mode must fail, not fall back to accepting everyone (cv-56 #12).
func TestNewRefusesUnknownMode(t *testing.T) {
	for _, mode := range []string{"", "oidc", "external"} {
		if a, err := New(Config{Mode: mode}); err == nil || a != nil {
			t.Errorf("mode %q: got %v, %v; want an error", mode, a, err)
		}
	}
}

func TestStaticAuth(t *testing.T) {
	a := mustNew(t, Config{Mode: "static", Keys: []StaticKey{
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

// The bearer scheme is case-insensitive (RFC 7235; cv-56 #12).
func TestBearerSchemeCaseInsensitive(t *testing.T) {
	a := mustNew(t, Config{Mode: "static", Keys: []StaticKey{{Key: "k1", User: "u"}}})
	for _, h := range []string{"Bearer k1", "bearer k1", "BEARER k1"} {
		r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		r.Header.Set("Authorization", h)
		if res, _ := a.Authenticate(r); res == nil || res.User != "u" {
			t.Errorf("%q: %+v", h, res)
		}
	}
	for _, h := range []string{"Bearer", "Bearer ", "Basic k1", "Bearerk1"} {
		r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		r.Header.Set("Authorization", h)
		if res, _ := a.Authenticate(r); res != nil {
			t.Errorf("%q accepted as %+v", h, res)
		}
	}
}

func TestNoAuthAcceptsMissingToken(t *testing.T) {
	r, err := mustNew(t, Config{Mode: "none"}).Authenticate(bearerReq(context.Background(), ""))
	if err != nil || r == nil || r.User != "anonymous" {
		t.Fatalf("%+v %v", r, err)
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
	a := mustNew(t, Config{Mode: "trusted_header", Secret: secret})
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
