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
  would cost, so no work of the same or lower priority is admitted on top of it.
- Waiters run by caller priority. Within a priority, those whose prompt is mostly cached go
  first; small prompts next; cold prompts last.

Measured on one 4x RTX 3090 box with 10 parallel coding agents: 91% of prompt tokens served
from cache and zero preemptions over 6.4 hours, against 13-16% with plain slot limits.

It also holds requests through engine restarts, sends SSE keepalives while a request waits,
logs one JSON line per request with token usage, and exports aggregate metrics.

## Quick start

You need one vLLM engine, version 0.24.0 or later (older versions encode KV events in a format
the proxy cannot read: it logs every batch as undecodable and predicts nothing cached), Go 1.23
(or Docker), and a model. The engine needs prefix caching,
KV cache events with a replay socket on the port after the event socket, and priority
scheduling (cavp sends a request's `priority` whenever it is not 0, which happens once a
caller has a priority or a request asks to run later; vLLM's default first-come scheduling
refuses a non-zero one). `--enable-prompt-tokens-details` makes the engine report cached tokens per
request, which the log and the hit-ratio metrics need. Set `--max-num-seqs` to the same value
as `admission.max_active` (left unset, `max_active` is 1: one request at a time). The engine must not require an API key. If it runs speculative
decoding (`--speculative-config`), set `admission.cache.speculative_decoding: true`: such an
engine serves a cached prefix minus its last block (its last unit with `--prefix-match-unit`),
and the cache mirror must know.

```sh
vllm serve Qwen/Qwen3-8B --port 8000 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --scheduling-policy priority \
  --max-num-seqs 8 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'
```

A minimal cavp config (`config.example.yaml` documents every setting):

```sh
cat > config.yaml <<'EOF'
upstream: "http://localhost:8000"
auth:
  mode: static
  keys:
    - key: ${CAVP_API_KEY}
      user: me
admission:
  max_active: 8
  cache:
    kv_events_endpoint: "tcp://localhost:5557"
EOF

export CAVP_API_KEY=$(openssl rand -hex 16)
go run ./cmd/cavp -check -config config.yaml   # validates and exits
go run ./cmd/cavp -config config.yaml          # listens on :4001 (`listen`)
```

The config is read once, at startup; a change needs a restart. The proxy serves plain HTTP:
put TLS in front of it if the network between clients and the proxy is not trusted.

Or with Docker: `docker build -t cavp .` and
`docker run --network host -e CAVP_API_KEY -v "$PWD/config.yaml:/etc/cavp/config.yaml:ro" cavp`
(host networking is Linux only; elsewhere publish port 4001 and point `upstream` at an
address the container can reach).

Once `curl localhost:4001/readyz` answers `ok`, send a request:

```sh
curl -N localhost:4001/v1/chat/completions \
  -H "Authorization: Bearer $CAVP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

Each API request logs one JSON line on stdout; `curl localhost:4001/metrics` shows, among
others, requests waiting for the engine (`cavp_ready_wait_requests`), for a tokenize slot
(`cavp_tokenize_wait_requests`) and for admission
(`cavp_queue_length`, `cavp_queue_oldest_wait_seconds`, the `cavp_queue_wait_seconds` histogram),
and failures to size prompts or read the engine's metrics (`cavp_tokenize_errors_total`,
`cavp_engine_metrics_errors_total`), and the KV event mirror (`cavp_kv_mirror_connected`,
`cavp_kv_mirror_blocks`, `cavp_kv_mirror_unsupported`, and `cavp_kv_mirror_events_total` by kind,
whose `skipped_*` kinds count what it could not use), requests by route and status
(`cavp_requests_total`), and the hit ratio (`cavp_cached_prompt_tokens_total` out of
`cavp_prompt_tokens_total`). Clients need no changes: on `/v1/chat/completions` a
conversation is identified by its first two messages, the model and the caller.

The KV event stream and its replay socket are required: the proxy rebuilds its copy of the
engine's cache from the replay whenever it (re)subscribes, and treats every prompt as uncached
until that replay completes. Requests wait until the engine's metrics have been read once (that
is where the KV pool size comes from). Engines that cache hybrid-model prefixes at a finer unit
than the block (`--prefix-match-unit`, as halt95's vLLM fork does) publish unit-sized block
events for where each prompt ends; the mirror matches them, taking the unit from the engine's
metrics (`prefix_match_unit`) or else from those events, so a returning turn is predicted
cached to the unit, not to the last full block. One engine means one engine: a data-parallel vLLM has a
KV pool and an event publisher per rank, and the proxy follows only one, so it is not
supported (`cavp -calibrate` refuses one).

## Requests

cavp serves a small, fixed set of routes; everything else is 404:

| Route | |
|---|---|
| `POST /v1/chat/completions`, `POST /v1/completions` | admitted, then forwarded |
| `POST /v1/tokenize`, `POST /tokenize` | forwarded to the engine's `/tokenize`, not admitted; shares the proxy's 4 tokenize slots (it waits for one) |
| `GET /v1/models` | the engine's model list, unchanged |
| `GET /healthz` | the process is up |
| `GET /readyz` | the engine is ready (see below) and its KV pool is known; else 503 with a plain-text reason |
| `GET /metrics` | aggregate state, Prometheus text format |

`/healthz`, `/readyz` and `/metrics` need no credentials, expose only totals and are not
logged. An unknown route is 404 before credentials are checked. Every route in the table
requires a caller, else 401 (`auth` in the config: `static` bearer keys, `trusted_header` behind
a gateway, or `none`; see `config.example.yaml`). Only `Content-Type`, `Accept`, `User-Agent`
and `X-Request-Id` are passed to the engine; the engine must not require an API key. Every API
request has an `X-Request-Id`: the caller's, else (absent or over 128 bytes) one the proxy
makes up. It is sent to the engine, echoed in the response and logged as `request_id`.

Errors the proxy itself returns have an OpenAI-shaped body,
`{"error":{"message":...,"type":"proxy_error","code":<status>}}`.

**Bodies** of the two admitted routes must be one JSON object (else 400). They are re-encoded
(key order and whitespace may change) and otherwise forwarded as sent, with three changes.
`priority` is set by the proxy: the caller's identity is the most urgent it may run, and an
integer `priority` field in the body may only make a request run later (clamped to -10..100);
the field is removed when the result is 0. On a streaming request,
`stream_options.include_usage` is set to `true` unless the request sets it: the final usage
chunk is how the proxy learns the size of a turn, which becomes its conversation's hold, so a
client that sets it `false` gets no hold. A request streams only when `stream` is the JSON value
`true`. `vllm_xargs.kv_cache_report_mode` is removed: a full report would corrupt the
proxy's mirror of the engine's cache. Bodies over `max_body_bytes` (256 MiB) are refused
with 413.

**Conversations** are identified by their first two messages, the model and the caller. Only
`/v1/chat/completions` has them: a `/v1/completions` request is charged while it runs and
leaves no hold. How long a finished turn's context is held depends on
`admission.cache.prefill_tokens_per_s` and `decode_interference` (see `config.example.yaml`),
and is at least 5 s.

**Waiting.** A request waits until the engine is ready and admission lets it in, however long
that takes; there are no timeouts and no queue limit, so a client that gives up simply leaves.
Only the two admitted routes wait: `/v1/models` and `/tokenize` are forwarded at once, and fail
with 502 while the engine is down. On shutdown (SIGTERM, SIGINT) waiting requests are answered
503 and running ones get 10 s to finish.
Streaming clients get SSE keepalive comments while they wait and whenever the stream is silent.
A client that stops reading for 5 minutes during a single write is disconnected, so it cannot
hold its place on the engine.

**Non-streaming requests** are charged their whole output cap (`max_completion_tokens`, else
`max_tokens`, else `admission.cache.default_output_tokens`) from admission, since their output is not seen until they finish. A client that sends a large
`max_tokens` (often the context length) is charged up to the whole pool: it still runs, but
nothing runs beside it. Prefer streaming, whose output is charged as it arrives, or a realistic
`max_tokens`. The cap is read as the engine reads it: `60000`, `60000.0`, `6e4` and `"60000"` are
the same cap, and `n` is read the same way. Several choices (`n`) or a batched `/v1/completions` prompt are several
sequences, each charged its own expected output.

**Engine restarts.** A readiness probe asks the engine's `/v1/models` every
`readiness.interval` (5 s); the engine is ready while it lists a model. While it is not,
admitted routes wait in the queue rather than fail. Nothing is retried: a request that fails
once it reached the engine returns its error. A stream has already answered 200, so its error
arrives as a final SSE error event (the engine's status as `code`) and `data: [DONE]`, after
the last complete event: an event the engine left half written is dropped, never relayed
(unless it grew past 1 MiB, after which bytes are relayed as they arrive). A running request
is given up when the probe has found the engine gone (connection refused, reset, no route, or
its name no longer resolves) for three intervals in a row and nothing came from the engine for
the request in that time, so a vanished engine cannot hold permits forever: with a 502 (an
error event on a stream), or, when a non-streaming response had already begun, by closing the
connection. A probe
that times out or gets an error answer never counts: a busy engine is alive, and a request on a
live engine runs as long as it takes. An engine wedged while still accepting connections is not
detected.

**Logs.** One JSON line per API request on stdout, with the outcome: for a stream the engine's
status, or 502 with an `error` when it failed or ended without `[DONE]`; `status` 499 for a
client that left before the end; token usage including `cached_prompt_tokens` (with
`--enable-prompt-tokens-details`; without it `usage_details` is absent and the hit-ratio
counters in `/metrics` stay at zero rather than report misses).

## Per-request overhead

Besides its prompt and output, every running request holds KV blocks of its own: recurrent
state on hybrid models, speculative-decoding scratch, prefix-cache checkpoints. How many depends
on the model and the engine version, from none on a plain transformer to dozens per request
on a hybrid model with speculative decoding. Admission charges each running request this
overhead, so it has to be right: too low and the engine runs out of KV space, evicts cached
contexts and queues or preempts requests; too high and capacity sits unused.

It is set in the config, per phase, for example:

```yaml
admission:
  cache:
    prefill_overhead_blocks: 12   # peak while a request prefills
    decode_overhead_blocks: 10    # while it decodes
```

Left unset, both are 28, the largest figure measured so far: safe on the engines we know, and
wasteful on a plain transformer, which needs close to none. An unset `decode_overhead_blocks`
takes the `prefill_overhead_blocks` figure; an unset `prefill_overhead_blocks` stays 28 whatever
decode is. Set both, or only the prefill one.

Output is charged as it streams: from the stream's own token counts when it reports usage on
every chunk (the client asked for `stream_options.continuous_usage_stats`), else as
`admission.cache.tokens_per_chunk` tokens per choice in each chunk (default 1; with speculative decoding
set 1 plus the draft length, e.g. 4 for MTP with 3 draft tokens).

### Measuring it

```sh
cavp -calibrate -config config.yaml
```

It loads the whole config (it must pass `-check`) but talks only to `upstream`. It sends 1, 2
and 4 concurrent requests straight to the engine and reads its KV usage, then prints the two
values with a margin of 2 blocks (the prefill figure never below the decode one), as an
`admission:` fragment to merge into your config, not to append. It needs the engine to itself:
stop the proxy (its traffic would make the rounds abort) and run it when nothing else reaches
the engine, ideally right after the engine starts. A round aborts rather than report a wrong
figure if the engine is not idle, other requests run, there is too little free room, or the
engine frees pinned blocks during it. Only the single-request round is required; a concurrent
round that aborts is printed as skipped. Calibration refuses an engine it cannot check: a
data-parallel one (more than one series per gauge), metrics without the KV usage, running or
waiting gauge, a usage gauge that never moved, or no token counts while streaming
(`stream_options.continuous_usage_stats`). Rerun it when
the model or the engine changes. It sends `VLLM_API_KEY` if set; the proxy itself sends no key.

Its prompts are `max(1536, 2 x block size + 256)` tokens, at most 16,384, with 512 output
tokens, and the prefill figure comes from a single request, so it is a floor: some engines take more per request on long prompts (on halt95's vLLM fork, v2.5,
with a hybrid model, 6-7 blocks on short prompts and up to 8.5 on 7,600-token ones). For
long-context traffic, set a figure above the calibrated one and watch the guard.

The proxy cannot measure this continuously. An engine that keeps finished turns' blocks pinned
for their next turn (halt95's vLLM fork does, from v2.5) reports most of its pool in use while nothing
runs, and it does not export how much of that is pinned.

### The overcommit guard

The guard is a safety net for a configured figure that is too low, or for anything else that
makes the engine hold more than admission expected. Its one signal is the engine preempting a
request (`vllm:num_preemptions_total` rising, per series). A read in which any series rose
raises a margin that is added to the overhead of every running and waiting request, by 2
blocks, at most once a minute (rises inside that minute add nothing) and up to 32 blocks. Each
15 minutes without a signal the margin falls by one step, so from the cap it takes 4 hours to
reach zero. Raises, lowerings and a signal at the cap are logged (`overcommit guard ...`), and
so is an engine whose metrics have no preemption counter; while `/metrics` cannot be read at
all the guard is blind and only `cavp_engine_metrics_errors_total` shows it.

Not signals: the engine evicting a context the proxy was holding (it drops cached prefixes for
its own reasons too; the hold is simply released and counted in
`cavp_cache_holds_evicted_total`, which also counts holds the mirror could never confirm), and the engine's waiting queue: vLLM counts
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

## License

Apache-2.0
