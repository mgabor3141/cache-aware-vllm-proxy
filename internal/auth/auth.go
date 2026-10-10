// Package auth identifies the caller of a request.
package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// Config selects how callers are authenticated.
//
//	none            every request is accepted as user "anonymous" at priority 0. For a
//	                proxy reachable only from trusted hosts.
//	static          bearer keys listed in the config, each with a user and a priority.
//	trusted_header  a gateway in front has authenticated the caller and names it in
//	                X-Cavp-User and X-Cavp-Priority, proving itself with X-Cavp-Secret
//	                (see trustedHeaderAuth). cavp must be reachable only from that gateway.
type Config struct {
	Mode   string      `yaml:"mode"`
	Keys   []StaticKey `yaml:"keys"`
	Secret string      `yaml:"secret"`
}

// StaticKey is one accepted bearer. Key usually comes from the environment
// (${MY_KEY}).
type StaticKey struct {
	Key      string `yaml:"key"`
	User     string `yaml:"user"`
	Priority int    `yaml:"priority"`
}

// Result is the caller's identity. User names the caller in logs and separates its
// sessions; Priority feeds admission (lower runs first, as in vLLM).
type Result struct {
	User     string
	Priority int
	Source   string // which mode accepted the caller
}

// Authenticator identifies the caller of a request. It returns (nil, nil) for an
// unknown or missing credential, and an error for a credential that is present but
// malformed (refused too, and worth logging).
type Authenticator interface {
	Authenticate(r *http.Request) (*Result, error)
}

// Bearer returns the request's bearer token, or "". The scheme is case-insensitive
// (RFC 7235).
func Bearer(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if len(h) > len("Bearer ") && strings.EqualFold(h[:len("Bearer ")], "Bearer ") {
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
	case "trusted_header":
		if len(c.Secret) < 16 {
			return fmt.Errorf("auth.mode trusted_header needs auth.secret of at least 16 characters (an unset ${VAR} expands to empty)")
		}
	case "":
		return fmt.Errorf("auth.mode is required: none, static or trusted_header")
	default:
		return fmt.Errorf("auth.mode %q: want none, static or trusted_header", c.Mode)
	}
	return nil
}

// New builds the configured mode. An invalid config is an error, never an open door.
func New(c Config) (Authenticator, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	switch c.Mode {
	case "static":
		return newStaticAuth(c.Keys), nil
	case "trusted_header":
		return &trustedHeaderAuth{secret: sha256.Sum256([]byte(c.Secret))}, nil
	}
	return noAuth{}, nil
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
