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

It also holds requests through engine restarts, sends SSE keepalives while a request waits,
logs one JSON line per request with token usage, and exports aggregate metrics.

## Quick start

```sh
vllm serve <model> --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'

cp config.example.yaml config.yaml   # set upstream and kv_events_endpoint
go run ./cmd/cavp -config config.yaml
```

The KV event stream and its replay socket are required: the proxy rebuilds its copy of the
engine's cache from the replay whenever it (re)subscribes, and treats every prompt as uncached
until that replay completes. Requests wait until the engine's metrics have been read once (that
is where the KV pool size comes from). One engine means one engine: a data-parallel vLLM has a
KV pool and an event publisher per rank, and the proxy follows only one, so it is not
supported (`cavp -calibrate` refuses one).

Clients need no changes: a conversation is identified by its first two messages.

## Requests

cavp serves a small, fixed set of routes; everything else is 404:

| Route | |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | admitted, then forwarded |
| `POST /v1/tokenize`, `POST /tokenize` | forwarded to the engine's `/tokenize`, not admitted |
| `GET /v1/models` | the engine's model list, unchanged |
| `GET /healthz` | the process is up |
| `GET /readyz` | the engine is ready (see below) |
| `GET /metrics` | aggregate state, Prometheus text format |

`/healthz`, `/readyz` and `/metrics` need no credentials and expose only totals. Every other
route requires a caller (`auth` in the config: `static` bearer keys, `trusted_header` behind a
gateway, or `none`). Only `Content-Type`, `Accept`, `User-Agent` and `X-Request-Id` are passed
to the engine; the engine must not require an API key.

**Bodies** are forwarded as sent, with two changes. `priority` is set by the proxy: the
caller's identity is the most urgent it may run, and a numeric `priority` field in the body
may only make a request run later (clamped to -10..100). On a streaming request,
`stream_options.include_usage` is set to `true` unless the request sets it: the final usage
chunk is how the proxy learns the size of a turn, which becomes its conversation's hold, so a
client that sets it `false` gets no hold. A request streams only when `stream` is the JSON value
`true`. Bodies over `max_body_bytes` (256 MiB) are refused with 413.

**Conversations** are identified by their first two messages, the model and the caller, so
clients need no changes.

**Waiting.** A request waits until the engine is ready and admission lets it in, however long
that takes; there are no timeouts and no queue limit, so a client that gives up simply leaves.
Streaming clients get SSE keepalive comments while they wait and whenever the stream is silent.
A client that stops reading for 5 minutes during a single write is disconnected, so it cannot
hold its place on the engine.

**Engine restarts.** A readiness probe asks the engine's `/v1/models` every
`readiness.interval` (5 s); the engine is ready while it lists a model. While it is not,
requests wait in the queue rather than fail. Nothing is retried: a request that fails once it
reached the engine returns its error (for a stream, as an SSE error event).

**Logs.** One JSON line per request on stdout, with the outcome: for a stream the engine's
status and an `error` when it failed or ended without `[DONE]`; `status` 499 for a client that
left while waiting; token usage including `cached_prompt_tokens`.

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

Output is charged as it streams: from the stream's own token counts when it reports usage on
every chunk, else as `tokens_per_chunk` tokens per chunk (default 1; with speculative decoding
set 1 plus the draft length, e.g. 4 for MTP with 3 draft tokens).

### Measuring it

```sh
cavp -calibrate -config config.yaml
```

This sends 1, 2 and 4 concurrent requests straight to the engine and reads its KV usage, then
prints the two values with a margin of 2 blocks. It needs the engine to itself: stop the proxy
(its guard would also react to the calibration traffic) and run it when nothing else reaches the
engine, ideally right after the engine starts. A round aborts rather than report a
wrong figure if other requests run or the engine frees pinned blocks during it, and calibration
refuses an engine it cannot check: a data-parallel one (more than one series per gauge), metrics
without the KV usage, running or waiting gauge, or a usage gauge that never moved. Rerun it when
the model or the engine changes. It sends `VLLM_API_KEY` if set; the proxy itself sends no key.

Its prompts span two blocks (at most 16K tokens), and the prefill figure comes from a single
request, so it is a floor: some engines take more per request on long prompts (on halt95's vLLM fork, v2.5,
with a hybrid model, 6-7 blocks on short prompts and up to 8.5 on 7,600-token ones). For
long-context traffic, set a figure above the calibrated one and watch the guard.

The proxy cannot measure this continuously. An engine that keeps finished turns' blocks pinned
for their next turn (halt95's vLLM fork does, from v2.5) reports most of its pool in use while nothing
runs, and it does not export how much of that is pinned.

### The overcommit guard

The guard is a safety net for a configured figure that is too low, or for anything else that
makes the engine hold more than admission expected. Its one signal is the engine preempting a
request (`vllm:num_preemptions_total` rising, per series). Each preemption raises a margin that
is added to the overhead of every running and waiting request, by 2 blocks, at most once a
minute and up to 32 blocks. After 15 minutes without a signal the margin falls by one step, back
to zero. Every change is logged (`overcommit guard raised the margin`, with its reason), and so
is an engine whose metrics have no preemption counter.

Not signals: the engine evicting a context the proxy was holding (it drops cached prefixes for
its own reasons too; the hold is simply released and counted in
`cavp_cache_holds_evicted_total`), and the engine's waiting queue: vLLM counts
every request it has not scheduled yet, including one admitted a moment ago and one waiting
behind another's prefill for the step's token budget, so it is non-zero in normal operation.

`/metrics` shows `cavp_guard_margin_blocks` and `cavp_guard_raises_total`.

The guard cannot tell a low overhead from other causes: a request that outruns its expected
output, or traffic that reaches the engine without going through the proxy, look the same. If
it raises the margin repeatedly, recalibrate or find the other cause; a margin that keeps
climbing means capacity is being lost.

## Layout

```
cmd/cavp/           the binary
internal/proxy/     flags and wiring (run.go), HTTP handling, admission (cache.go,
                    admission.go, guard.go), readiness, streaming, usage accounting,
                    metrics, calibration
internal/auth/      none / static / trusted_header authentication
kvmirror/           a mirror of a vLLM engine's prefix cache from its KV events; usable
                    on its own
docs/design.md      why it works the way it does, with the measurements
```
