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
	if r, _ := a.Authenticate(ctx, "batch-key"); r == nil || r.User != "batch" || r.Priority != 10 || r.Source != "static" {
		t.Fatalf("batch: %+v", r)
	}
	for _, tok := range []string{"", "alice-ke", "alice-key2", "nope"} {
		if r, _ := a.Authenticate(ctx, tok); r != nil {
			t.Errorf("token %q accepted as %+v", tok, r)
		}
	}
}

func TestNoAuthAcceptsMissingToken(t *testing.T) {
	r, err := New(Config{Mode: "none"}).Authenticate(context.Background(), "")
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
	if r, err := a.Authenticate(ctx, "ok"); err != nil || r == nil || r.User != "u" || r.Priority != 5 || r.Source != "external" {
		t.Fatalf("ok: %+v %v", r, err)
	}
	if r, err := a.Authenticate(ctx, "bad"); r != nil || err != nil {
		t.Fatalf("bad: %+v %v", r, err)
	}
	_, err := a.Authenticate(ctx, "slow")
	if rl, ok := err.(*RateLimitedError); !ok || rl.RetryAfter != 7 {
		t.Fatalf("slow: %v", err)
	}
	for _, tok := range []string{"anonymous", "badprio"} {
		if r, err := a.Authenticate(ctx, tok); r != nil || err == nil {
			t.Errorf("%s: accepted as %+v (%v)", tok, r, err)
		}
	}
	if r, err := a.Authenticate(ctx, "broken"); r != nil || err == nil {
		t.Fatalf("broken: %+v %v", r, err)
	}
	if r, err := a.Authenticate(ctx, ""); r != nil || err != nil {
		t.Fatalf("missing token reached the backend: %+v %v", r, err)
	}
}
