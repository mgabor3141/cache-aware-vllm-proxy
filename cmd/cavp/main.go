// Command cavp is cache-aware-vllm-proxy: an OpenAI-compatible proxy that admits
// requests to one vLLM engine by what its prefix cache holds.
package main

import "github.com/mgabor3141/cache-aware-vllm-proxy/internal/proxy"

func main() { proxy.Main() }
