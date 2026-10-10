package kvmirror_test

import (
	"context"
	"fmt"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

// A mirror subscribes to the engine's KV event stream (vLLM --kv-events-config with a ZMQ
// publisher, replay on the next port) and answers how much of a tokenized prompt the
// engine has cached. Until it is subscribed (Healthy), it knows nothing: treat every
// prompt as uncached.
func Example() {
	m := kvmirror.New()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	endpoint := "tcp://vllm:5557"
	go m.Run(ctx, endpoint, kvmirror.ReplayEndpoint(endpoint))

	for deadline := time.Now().Add(5 * time.Second); !m.Healthy() && time.Now().Before(deadline); {
		time.Sleep(100 * time.Millisecond)
	}
	prompt := []int64{151644, 872, 198} // token ids, e.g. from vLLM's /tokenize
	if m.Healthy() {
		fmt.Println(m.CachedTokens(prompt), "tokens cached")
	}
}
