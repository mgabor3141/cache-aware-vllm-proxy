package proxy

import (
	"os"
	"strings"
	"testing"
)

// The example config must load: it is what new users start from.
func TestShippedConfigParses(t *testing.T) {
	t.Setenv("CAVP_API_KEY", "test-key")
	if _, err := LoadConfig("../../config.example.yaml"); err != nil {
		t.Fatalf("config.example.yaml does not load: %v", err)
	}
}

// A recovery probe slower than the recovery budget could only notice recovery after the
// requests waiting on it had failed.
func TestProbeSlowerThanRecoveryFails(t *testing.T) {
	f := t.TempDir() + "/c.yaml"
	os.WriteFile(f, []byte("upstream: http://x\nauth:\n  mode: none\nretry:\n  recovery_timeout: 10s\n  recovery_probe_interval: 30s\n"), 0o600)
	if _, err := LoadConfig(f); err == nil || !strings.Contains(err.Error(), "recovery_probe_interval") {
		t.Fatalf("accepted: %v", err)
	}
}

func TestUnknownConfigKeyFails(t *testing.T) {
	f := t.TempDir() + "/c.yaml"
	os.WriteFile(f, []byte("upstream: http://x\nauth:\n  mode: none\ntts_upstream: http://y\n"), 0o600)
	if _, err := LoadConfig(f); err == nil || !strings.Contains(err.Error(), "tts_upstream") {
		t.Fatalf("retired key accepted: %v", err)
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
		f := t.TempDir() + "/c.yaml"
		os.WriteFile(f, []byte("upstream: http://x\nauth:\n  mode: none\nadmission:\n  cache:\n    enabled: true\n"+c.yaml), 0o600)
		cfg, err := LoadConfig(f)
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
