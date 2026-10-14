package proxy

import (
	"encoding/json"
	"testing"
)

// A request the engine never answered gets a 502 in the proxy's error envelope, whose
// message names what failed: the model list, a tokenization or an inference request.
func TestUpstreamFailureMessagePerRoute(t *testing.T) {
	h := newHarness(t, nil, nil)
	h.eng.Close()
	for _, c := range []struct{ method, path, body, want string }{
		{"GET", "/v1/models", "", "model list upstream request failed"},
		{"POST", "/v1/tokenize", `{"model":"m","prompt":"hi"}`, "tokenize upstream request failed"},
		{"POST", "/v1/chat/completions", chatBody, "inference upstream request failed"},
	} {
		status, body := h.do(c.method, c.path, c.body, nil)
		var env struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    int    `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(body), &env); err != nil {
			t.Fatalf("%s: body %q: %v", c.path, body, err)
		}
		if status != 502 || env.Error.Code != 502 || env.Error.Type != "proxy_error" || env.Error.Message != c.want {
			t.Errorf("%s: %d %+v, want 502 %q", c.path, status, env.Error, c.want)
		}
	}
}
