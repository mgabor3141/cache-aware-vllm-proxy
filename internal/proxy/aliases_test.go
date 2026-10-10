package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModelResolverRefreshCapturesContextWindow(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"qwen-test","max_model_len":196608}]}`))
	}))
	defer server.Close()

	resolver := NewModelResolver(server.URL, server.Client())
	if err := resolver.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := resolver.Current(); got != "qwen-test" {
		t.Fatalf("Current() = %q, want qwen-test", got)
	}
	if got := resolver.MaxModelLen(); got != 196608 {
		t.Fatalf("MaxModelLen() = %d, want 196608", got)
	}
}
