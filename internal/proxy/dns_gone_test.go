package proxy

import (
	"context"
	"net"
	"net/url"
	"testing"
	"time"
)

// An engine whose name no longer resolves is gone, like one refusing connections; a
// temporary resolver failure is not.
func TestEngineGoneWhenNameDoesNotResolve(t *testing.T) {
	notFound := &url.Error{Op: "Get", URL: "http://vllm:8000/v1/models", Err: &net.OpError{
		Op: "dial", Net: "tcp", Err: &net.DNSError{Err: "no such host", Name: "vllm", IsNotFound: true}}}
	if !engineGone(notFound) {
		t.Errorf("no such host not counted as gone")
	}
	if engineGone(&net.DNSError{Err: "server misbehaving", Name: "vllm", IsTemporary: true}) {
		t.Errorf("a temporary resolver failure counted as gone")
	}
	h := newReadiness("http://nosuchhost.invalid:8000", time.Second)
	if err := h.probe(context.Background()); err == nil || !engineGone(err) {
		t.Errorf("probe of an unresolvable name: %v, gone=%v", err, err != nil && engineGone(err))
	}
}
