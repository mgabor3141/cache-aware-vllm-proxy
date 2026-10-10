// Config types and loader. One YAML file, read at startup; changes need a restart.
package proxy

import (
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/internal/auth"
	"gopkg.in/yaml.v3"
)

// Config is the entire on-disk configuration.
type Config struct {
	// Listen is the address the HTTP server binds to (plain HTTP; put TLS in front if
	// it is exposed).
	Listen string `yaml:"listen"`

	// Upstream is the vLLM base URL. One engine: cache-aware admission models a single
	// KV pool.
	Upstream string `yaml:"upstream"`

	// MaxBodyBytes caps a request body. A larger one is answered 413 without being
	// read further. Default 256 MiB: multi-image threads are tens of megabytes.
	MaxBodyBytes int64 `yaml:"max_body_bytes"`

	// Readiness controls the engine readiness probe (health.go).
	Readiness ReadinessConfig `yaml:"readiness"`

	// Admission bounds what runs on the engine and queues the rest (admission.go,
	// cache.go).
	Admission AdmissionConfig `yaml:"admission"`

	// Auth selects how callers are authenticated (internal/auth).
	Auth auth.Config `yaml:"auth"`
}

// ReadinessConfig: the engine is ready while GET /v1/models lists a model. The probe
// runs every Interval; while it fails, admitted requests wait.
type ReadinessConfig struct {
	Interval time.Duration `yaml:"interval"`
}

const defaultMaxBodyBytes = 256 << 20

// LoadConfig reads and validates a config file.
//
// ${VAR} references inside values are replaced from the environment after the YAML is
// parsed, so a value can never become structure: a secret containing a newline, a
// colon or " #" stays one string. Unknown keys are errors, so a misspelt or retired
// setting fails at startup instead of being ignored.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	expandEnv(&doc)
	c := &Config{}
	if len(doc.Content) > 0 {
		// Decoding the node is strict too: re-encode it and decode with KnownFields,
		// which yaml.Node.Decode does not offer.
		out, err := yaml.Marshal(&doc)
		if err != nil {
			return nil, fmt.Errorf("parse config %s: %w", path, err)
		}
		dec := yaml.NewDecoder(strings.NewReader(string(out)))
		dec.KnownFields(true)
		if err := dec.Decode(c); err != nil {
			return nil, fmt.Errorf("parse config %s: %w", path, err)
		}
	}
	return c, c.validate()
}

// envRef is a ${VAR} reference. A bare $ is left alone, so a literal "abc$def" survives.
var envRef = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// expandEnv replaces ${VAR} in every scalar value (not keys). A substituted scalar loses
// its parsed tag and style, so the encoder re-emits it as one scalar of whatever type it
// now reads as (a number for max_active: ${N}), quoted or block-styled as needed.
func expandEnv(n *yaml.Node) {
	switch n.Kind {
	case yaml.ScalarNode:
		if v := envRef.ReplaceAllStringFunc(n.Value, func(m string) string {
			return os.Getenv(m[2 : len(m)-1])
		}); v != n.Value {
			n.Value, n.Tag, n.Style = v, "", 0
		}
	case yaml.MappingNode:
		for i := 1; i < len(n.Content); i += 2 {
			expandEnv(n.Content[i])
		}
	default:
		for _, c := range n.Content {
			expandEnv(c)
		}
	}
}

func (c *Config) validate() error {
	if c.Listen == "" {
		c.Listen = ":4001"
	}
	c.Upstream = strings.TrimRight(c.Upstream, "/")
	if c.Upstream == "" {
		return fmt.Errorf("upstream is required")
	}
	if u, err := url.Parse(c.Upstream); err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return fmt.Errorf("upstream %q: want an http(s) URL with a host, e.g. http://vllm:8000", c.Upstream)
	}
	if err := c.Auth.Validate(); err != nil {
		return err
	}
	if c.MaxBodyBytes < 0 {
		return fmt.Errorf("max_body_bytes must be positive")
	}
	if c.MaxBodyBytes == 0 {
		c.MaxBodyBytes = defaultMaxBodyBytes
	}
	if c.Readiness.Interval < 0 {
		return fmt.Errorf("readiness.interval must be positive")
	}
	if c.Readiness.Interval == 0 {
		c.Readiness.Interval = 5 * time.Second
	}
	c.Admission.applyDefaults()
	return c.Admission.validate()
}
