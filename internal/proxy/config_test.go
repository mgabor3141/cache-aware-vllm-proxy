package proxy

import (
	"os"
	"strings"
	"testing"
	"time"
)

func loadYAML(t *testing.T, yaml string) (*Config, error) {
	t.Helper()
	f := t.TempDir() + "/c.yaml"
	if err := os.WriteFile(f, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return LoadConfig(f)
}

const minimalYAML = "upstream: http://x\nauth:\n  mode: none\n"

// The example config must load: it is what new users start from.
func TestShippedConfigParses(t *testing.T) {
	t.Setenv("CAVP_API_KEY", "test-key")
	if _, err := LoadConfig("../../config.example.yaml"); err != nil {
		t.Fatalf("config.example.yaml does not load: %v", err)
	}
}

func TestDefaults(t *testing.T) {
	cfg, err := loadYAML(t, minimalYAML)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.MaxBodyBytes != 256<<20 || cfg.Readiness.Interval != 5*time.Second || cfg.Listen != ":4001" {
		t.Fatalf("%+v", cfg)
	}
}

// Unknown keys fail, including every key this version removed.
func TestUnknownAndRetiredKeysFail(t *testing.T) {
	for _, key := range []string{
		"tts_upstream: x", "aliases: {}", "retry: {}", "models_refresh_interval: 1m",
		"max_inflight_body_bytes: 1", "body_amplification: 2", "max_chunked_body_bytes: 1",
		"sampling_defaults: {}", "reasoning_effort_mappings: {}", "reasoning_effort_defaults: {}",
	} {
		name := strings.SplitN(key, ":", 2)[0]
		if _, err := loadYAML(t, minimalYAML+key+"\n"); err == nil || !strings.Contains(err.Error(), name) {
			t.Errorf("%s accepted: %v", name, err)
		}
	}
	if _, err := loadYAML(t, "upstream: http://x\nauth:\n  mode: external\n  url: http://a\n"); err == nil {
		t.Error("auth.mode external accepted")
	}
}

func TestValidation(t *testing.T) {
	for _, c := range []struct{ yaml, bad string }{
		{"upstream: vllm:8000\nauth:\n  mode: none\n", "upstream"},
		{"upstream: http://\nauth:\n  mode: none\n", "upstream"},
		{"auth:\n  mode: none\n", "upstream"},
		{minimalYAML + "max_body_bytes: -1\n", "max_body_bytes"},
		{minimalYAML + "readiness:\n  interval: -1s\n", "readiness"},
	} {
		if _, err := loadYAML(t, c.yaml); err == nil || !strings.Contains(err.Error(), c.bad) {
			t.Errorf("%q: %v, want an error naming %s", c.yaml, err, c.bad)
		}
	}
}

// ${VAR} is substituted into values after parsing, so a value cannot become structure,
// and text that is not a ${VAR} reference is left alone (cv-57 #2).
func TestEnvExpansionIsValueOnly(t *testing.T) {
	t.Setenv("KEY_WITH_NEWLINE", "real\n    - key: injected\n      user: attacker")
	t.Setenv("KEY_WITH_HASH", "abc #def")
	t.Setenv("MAX_ACTIVE", "6")
	cfg, err := loadYAML(t, `upstream: http://x
admission:
  max_active: ${MAX_ACTIVE}
auth:
  mode: static
  keys:
    - key: ${KEY_WITH_NEWLINE}
      user: one
    - key: "${KEY_WITH_HASH}"
      user: two
    - key: abc$def
      user: three
`)
	if err != nil {
		t.Fatal(err)
	}
	keys := cfg.Auth.Keys
	if len(keys) != 3 {
		t.Fatalf("keys: %+v", keys)
	}
	if keys[0].Key != "real\n    - key: injected\n      user: attacker" || keys[1].Key != "abc #def" || keys[2].Key != "abc$def" {
		t.Fatalf("keys: %+v", keys)
	}
	if cfg.Admission.MaxActive != 6 {
		t.Fatalf("max_active %d", cfg.Admission.MaxActive)
	}
	// An unset variable is empty, which validation then refuses where it matters.
	if _, err := loadYAML(t, "upstream: http://x\nauth:\n  mode: static\n  keys:\n    - key: ${CAVP_UNSET_FOR_TEST}\n      user: u\n"); err == nil {
		t.Fatal("empty key accepted")
	}
}

func TestCacheConfigValidation(t *testing.T) {
	for _, c := range []struct {
		yaml string
		bad  string // substring of the error, "" when valid
		pre  int    // expected prefill overhead when valid
	}{
		{"", "", defaultOverheadBlocks},
		{"    prefill_overhead_blocks: 0\n", "", 0},
		{"    prefill_overhead_blocks: -1\n", "prefill_overhead_blocks", 0},
		{"    decode_interference: 1.5\n", "decode_interference", 0},
	} {
		cfg, err := loadYAML(t, minimalYAML+"admission:\n  cache:\n    enabled: true\n"+c.yaml)
		if c.bad != "" {
			if err == nil || !strings.Contains(err.Error(), c.bad) {
				t.Errorf("%q: err %v, want one naming %s", c.yaml, err, c.bad)
			}
			continue
		}
		if err != nil {
			t.Errorf("%q: %v", c.yaml, err)
			continue
		}
		if got := cfg.Admission.Cache.overhead(false); got != c.pre {
			t.Errorf("%q: prefill overhead %d, want %d", c.yaml, got, c.pre)
		}
	}
}
