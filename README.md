# cache-aware-vllm-proxy

An OpenAI-compatible reverse proxy that decides **when** each request reaches a single vLLM
engine, so that many concurrent agents keep their prefix cache.

> Status: young. It runs in production against one engine, and the configuration may still
> change between versions.

## Why

Coding agents run tool loops on long contexts: each turn adds a few hundred tokens to 50-250K
of context and comes back within seconds. If the engine still has that context cached, the
turn costs a short prefill. If not, the whole context is recomputed, which takes tens of
seconds and slows every other request.

With many agents, vLLM evicts contexts that are about to be reused, because it sees requests,
not sessions. This proxy keeps the engine from overfilling and keeps a returning agent's
context resident:

- It mirrors the engine's prefix cache from vLLM's KV cache events, so it knows how much of
  each prompt is cached before admitting it.
- It admits only what fits in the engine's KV pool, read from the engine's metrics, and
  queues the rest, so the engine has no reason to evict a context in use or preempt a request.
- When an agent's turn ends, its context is held for a time that scales with what a miss
  would cost, so nothing new is admitted on top of it.
- Waiters whose prompt is mostly cached go first; small prompts next; cold prompts last.

Measured on one 4x RTX 3090 box with 10 parallel coding agents: 91% of prompt tokens served
from cache and zero preemptions over 6.4 hours, against 13-16% with plain slot limits.

It also retries engine restarts transparently, sends SSE keepalives while a request waits,
and logs one JSON line per request with token usage.

## Quick start

```sh
vllm serve <model> --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'

cp config.example.yaml config.yaml   # set upstream and kv_events_endpoint
go run ./cmd/cavp -config config.yaml
```

Clients need no changes: a conversation is identified by its first two messages. A client
may send `X-Session-Key` to name its conversation explicitly.

## License

Apache-2.0

## Per-request overhead

Besides its prompt and output, every running request holds KV blocks of its own: recurrent
state on hybrid models, speculative-decoding scratch, prefix-cache checkpoints. How many depends
on the model and the engine version, from none on a plain transformer to dozens per request
on a hybrid model with speculative decoding. Admission charges each running request this
overhead, so it has to be right: too low and the engine runs out of KV space, evicts cached
contexts and queues or preempts requests; too high and capacity sits unused.

It is set in the config, per phase:

```yaml
admission:
  cache:
    prefill_overhead_blocks: 12   # peak while a request prefills
    decode_overhead_blocks: 10    # while it decodes
```

### Measuring it

```sh
cavp -calibrate -config config.yaml
```

This sends 1, 2 and 4 concurrent requests straight to the engine and reads its KV usage, then
prints the two values with a margin of 2 blocks. It needs the engine to itself: stop the proxy
(its guard would also react to the calibration traffic) and run it when nothing else reaches the
engine, ideally right after the engine starts. A round aborts rather than report a
wrong figure if other requests run or the engine frees pinned blocks during it. Rerun it when
the model or the engine changes. Set `VLLM_API_KEY` if the engine requires a key.

Its prompts span two blocks (at most 16K tokens), and the prefill figure comes from a single
request, so it is a floor: some engines take more per request on long prompts (on halt95's vLLM fork, v2.5,
with a hybrid model, 6-7 blocks on short prompts and up to 8.5 on 7,600-token ones). For
long-context traffic, set a figure above the calibrated one and watch the guard.

The proxy cannot measure this continuously. An engine that keeps finished turns' blocks pinned
for their next turn (halt95's vLLM fork does, from v2.5) reports most of its pool in use while nothing
runs, and it does not export how much of that is pinned.

### The overcommit guard

The guard is a safety net for a configured figure that is too low, or for anything else that
makes the engine hold more than admission expected. Two signals raise a margin that is added to
every running request's overhead:

- **early:** the engine evicted a cached context the proxy was still protecting (seen through
  KV events) while the proxy's accounting was near its budget. Evictions with plenty of room
  left are the engine's own policy (it may keep only so many finished turns) and do not count;
- **late:** the engine preempted a request.

Each raises the margin by 2 blocks, at most once a minute and up to 32 blocks. After 15
minutes without a signal the margin falls by one step, back to zero. Every change is logged
(`overcommit guard raised the margin`, with its reason), and `/debug/residency` shows
`guard_margin_blocks` and `guard_raises_total`.

The engine's waiting queue is deliberately not a signal: vLLM counts every request it has not
scheduled yet, including one admitted a moment ago and one waiting behind another's prefill
for the step's token budget, so it is non-zero in normal operation.

The guard cannot tell a low overhead from other causes: a request that outruns its expected
output, or traffic that reaches the engine without going through the proxy, look the same. If
it raises the margin repeatedly, recalibrate or find the other cause; a margin that keeps
climbing means capacity is being lost.

## Layout

```
cmd/cavp/           the binary
internal/proxy/     flags and wiring (run.go), HTTP handling, admission (cache.go,
                    admission.go, guard.go), retry, streaming, usage accounting, calibration
internal/auth/      none / static / trusted_header authentication
kvmirror/           a mirror of a vLLM engine's prefix cache from its KV events; usable
                    on its own
docs/design.md      why it works the way it does, with the measurements
```
