package proxy

import (
	"bytes"
	"encoding/json"
	"testing"
)

// An output cap the engine accepts is priced whatever its spelling: an integral float or
// a numeric string reaches the engine as that integer (l3-94 #2).
func TestMaxTokensSpellings(t *testing.T) {
	for body, want := range map[string]int{
		`{"max_tokens":60000}`: 60000, `{"max_tokens":60000.0}`: 60000, `{"max_tokens":6e4}`: 60000,
		`{"max_tokens":"60000"}`: 60000, `{"max_completion_tokens":6e4}`: 60000,
		`{"max_tokens":0.5}`: 0, `{"max_tokens":-60000}`: 0, `{"max_tokens":"lots"}`: 0, `{}`: 0,
	} {
		var m map[string]any
		d := json.NewDecoder(bytes.NewReader([]byte(body)))
		d.UseNumber()
		if err := d.Decode(&m); err != nil {
			t.Fatal(err)
		}
		if got := maxTokensOf(m); got != want {
			t.Errorf("%s: cap %d, want %d", body, got, want)
		}
	}
}
