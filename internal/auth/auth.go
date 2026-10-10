package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// Config selects how callers are authenticated.
//
//	none      every request is accepted as user "anonymous" at priority 0. For a
//	          proxy reachable only from trusted hosts.
//	static    bearer keys listed in the config, each with a user and a priority.
//	external        each bearer is validated by POST <url>/validate (see externalAuth).
//	trusted_header  a gateway in front has authenticated the caller and names it in
//	                X-Cavp-User and X-Cavp-Priority, proving itself with X-Cavp-Secret
//	                (see trustedHeaderAuth). cavp must be reachable only from that gateway.
type Config struct {
	Mode   string      `yaml:"mode"`
	URL    string      `yaml:"url"`
	Keys   []StaticKey `yaml:"keys"`
	Secret string      `yaml:"secret"`
}

// StaticKey is one accepted bearer. Key usually comes from the environment
// (${MY_KEY}): the config is expanded before parsing.
type StaticKey struct {
	Key      string `yaml:"key"`
	User     string `yaml:"user"`
	Priority int    `yaml:"priority"`
}

// Result is the caller's identity. User and Sub name the caller in logs and
// separate their sessions; Priority feeds admission (lower runs first, as in vLLM).
type Result struct {
	User     string
	Priority int
	Source   string // which mode or backend accepted the caller
	Sub      string // stable subject id when the backend has one
}

// Authenticator identifies the caller of a request. It returns (nil, nil) for an
// unknown or missing credential, and a *RateLimitedError when the backend
// rate-limited the caller.
type Authenticator interface {
	Authenticate(r *http.Request) (*Result, error)
}

// Bearer returns the request's bearer token, or "".
func Bearer(r *http.Request) string {
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
		return strings.TrimSpace(h[len("Bearer "):])
	}
	return ""
}

// Header names of the trusted_header mode.
const (
	HeaderUser     = "X-Cavp-User"
	HeaderPriority = "X-Cavp-Priority"
	HeaderSecret   = "X-Cavp-Secret"
)

// RateLimitedError is returned verbatim to the client (status and Retry-After).
type RateLimitedError struct {
	Status     int
	RetryAfter int // seconds; 0 if none provided
}

func (e *RateLimitedError) Error() string {
	return "rate limited"
}

func (c Config) Validate() error {
	switch c.Mode {
	case "none":
	case "static":
		if len(c.Keys) == 0 {
			return fmt.Errorf("auth.mode static needs auth.keys")
		}
		seen := map[string]bool{}
		for i, k := range c.Keys {
			if k.Key == "" || k.User == "" {
				return fmt.Errorf("auth.keys[%d]: key and user are required (an unset ${VAR} expands to empty)", i)
			}
			if seen[k.Key] {
				return fmt.Errorf("auth.keys[%d] (%s): duplicate key", i, k.User)
			}
			seen[k.Key] = true
		}
	case "external":
		if c.URL == "" {
			return fmt.Errorf("auth.mode external needs auth.url")
		}
	case "trusted_header":
		if len(c.Secret) < 16 {
			return fmt.Errorf("auth.mode trusted_header needs auth.secret of at least 16 characters (an unset ${VAR} expands to empty)")
		}
	case "":
		return fmt.Errorf("auth.mode is required: none, static, external or trusted_header")
	default:
		return fmt.Errorf("auth.mode %q: want none, static, external or trusted_header", c.Mode)
	}
	return nil
}

// New builds the configured mode. The config must have validated.
func New(c Config) Authenticator {
	switch c.Mode {
	case "static":
		return newStaticAuth(c.Keys)
	case "external":
		return newExternalAuth(c.URL)
	case "trusted_header":
		return &trustedHeaderAuth{secret: sha256.Sum256([]byte(c.Secret))}
	default:
		return noAuth{}
	}
}

type noAuth struct{}

func (noAuth) Authenticate(*http.Request) (*Result, error) {
	return &Result{User: "anonymous", Source: "none"}, nil
}

// staticAuth compares a token's sha256 against each key's in constant time, so
// the time taken says nothing about how much of a key matched.
type staticAuth struct {
	keys []staticEntry
}

type staticEntry struct {
	sum [32]byte
	res Result
}

func newStaticAuth(keys []StaticKey) *staticAuth {
	a := &staticAuth{}
	for _, k := range keys {
		a.keys = append(a.keys, staticEntry{sha256.Sum256([]byte(k.Key)),
			Result{User: k.User, Priority: k.Priority, Source: "static"}})
	}
	return a
}

func (a *staticAuth) Authenticate(r *http.Request) (*Result, error) {
	token := Bearer(r)
	if token == "" {
		return nil, nil
	}
	sum := sha256.Sum256([]byte(token))
	var found *Result
	for i := range a.keys {
		if subtle.ConstantTimeCompare(sum[:], a.keys[i].sum[:]) == 1 {
			r := a.keys[i].res
			found = &r
		}
	}
	return found, nil
}

// externalAuth delegates to a validation service: POST <url>/validate with the
// caller's Authorization header.
//
//	204  accepted; identity from X-Authenticated-User (required), X-User-Sub,
//	     X-User-Priority (an integer) and X-Auth-Source response headers
//	401  unknown token
//	429  rate limited; status and Retry-After are passed to the client
//
// Anything else, or no answer within 2 s, is treated as an unknown token (and
// logged), so an outage of the service fails closed.
type externalAuth struct {
	baseURL string
	http    *http.Client
}

func newExternalAuth(baseURL string) *externalAuth {
	return &externalAuth{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 2 * time.Second},
	}
}

func (c *externalAuth) Authenticate(r *http.Request) (*Result, error) {
	token := Bearer(r)
	if token == "" {
		return nil, nil
	}
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, c.baseURL+"/validate", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusNoContent:
		// An accepted caller must be named, and a priority the backend sent must be
		// read as sent: otherwise fail closed rather than run it as anonymous or as 0.
		user := resp.Header.Get("X-Authenticated-User")
		if user == "" {
			return nil, fmt.Errorf("auth: 204 without X-Authenticated-User")
		}
		priority := 0
		if p := resp.Header.Get("X-User-Priority"); p != "" {
			var err error
			if priority, err = strconv.Atoi(p); err != nil {
				return nil, fmt.Errorf("auth: X-User-Priority %q is not a number", p)
			}
		}
		src := resp.Header.Get("X-Auth-Source")
		if src == "" {
			src = "external"
		}
		return &Result{
			User:     user,
			Sub:      resp.Header.Get("X-User-Sub"),
			Priority: priority,
			Source:   src,
		}, nil
	case http.StatusUnauthorized:
		return nil, nil
	case http.StatusTooManyRequests:
		ra, _ := strconv.Atoi(resp.Header.Get("Retry-After"))
		return nil, &RateLimitedError{Status: http.StatusTooManyRequests, RetryAfter: ra}
	default:
		return nil, fmt.Errorf("auth: unexpected status %d", resp.StatusCode)
	}
}

// trustedHeaderAuth accepts the identity a gateway put in the request headers. The
// secret (compared by hash, in constant time) shows the headers came from the gateway
// and not from a caller who reached cavp directly; without it, or without a user, the
// request is refused. The priority header is optional (0) but must be an integer.
type trustedHeaderAuth struct {
	secret [32]byte
}

func (a *trustedHeaderAuth) Authenticate(r *http.Request) (*Result, error) {
	sum := sha256.Sum256([]byte(r.Header.Get(HeaderSecret)))
	if subtle.ConstantTimeCompare(sum[:], a.secret[:]) != 1 {
		return nil, nil
	}
	user := r.Header.Get(HeaderUser)
	if user == "" {
		return nil, fmt.Errorf("auth: trusted request without %s", HeaderUser)
	}
	priority := 0
	if p := r.Header.Get(HeaderPriority); p != "" {
		var err error
		if priority, err = strconv.Atoi(p); err != nil {
			return nil, fmt.Errorf("auth: %s %q is not a number", HeaderPriority, p)
		}
	}
	return &Result{User: user, Priority: priority, Source: "trusted_header"}, nil
}
