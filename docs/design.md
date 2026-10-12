# Design

cache-aware-vllm-proxy sits in front of one vLLM engine and decides when each request reaches
it. Everything here follows from one workload: many coding agents running tool loops on long
contexts against a single GPU box. This document explains the choices and the measurements
behind them.

## The problem

An agent turn adds a few hundred to a few thousand tokens to 50-250K tokens of context, and the
agent comes back within seconds (in one measured run of review agents, 91% of next turns arrived
within 1 s of the previous answer and all within 5 s). If the engine still has the context in
its prefix cache, the turn costs a short prefill. If not, the whole context is recomputed: tens
of seconds at a few thousand tokens per second, while every other request's decoding slows down
too (a running prefill cut decode speed by about 94% on the reference engine).

vLLM sees requests, not conversations. Under load it evicts a context that is about to be reused,
because an agent between turns looks like an idle cached prefix. With 20 agents on a 640K-token
pool the starting point was 13-16% of prompt tokens served from cache, and the engine spent most
of its time recomputing.

## Principles

- **All queueing happens in the proxy.** The engine is only sent what it can run. A request
  queued or preempted inside the engine loses its cache and blocks others; one waiting in the
  proxy costs nothing.
- **No timeouts and no queue limit.** Agents retry. Timeouts that fire for many waiters at once
  only add load. A retry from the same conversation (within 5 min) keeps the place
  its previous attempt had on arrival; giving up and returning again does not add up.
  The one bound is on a dead engine, not on a request: a running request is given up once
  the readiness probe has found the engine gone (refused, reset, no route, name not found; never a mere timeout)
  for three intervals and the engine sent it nothing meanwhile.
- **One job.** The proxy decides when a chat request reaches the engine, and nothing else: no
  model aliases, no request rewriting beyond priority and usage reporting, no replays. Deployment
  concerns (external identity, rate limits, CORS) belong to a gateway in front, which names the
  caller through trusted headers.
- **Judge by real traffic.** An early simulator gave answers that did not survive contact with
  real agents; every figure here comes from live runs.
- **Depend only on public engine interfaces:** the OpenAI API, `/metrics`, and KV cache events.
  The proxy survived two engine upgrades without code changes for that reason.

## Knowing what is cached: the KV mirror

The engine publishes every prefix-cache block it stores or evicts, and an event when it clears
them all (`--kv-events-config`, ZMQ). The `kvmirror` package replays that stream into a copy of
the cache, keyed by a hash chained over each block's token IDs and its `extra_keys` (images,
LoRA, cache salt), so for any prompt the proxy can say how many leading tokens the engine would
serve from cache. Token IDs alone never match a block stored with extra keys, and a block whose
parent the mirror never saw is skipped (which only under-reports).
Every admitted request is tokenized on arrival (`/tokenize`, with the chat template on the chat
route) to get those tokens. When that fails, the prompt is priced from the body: 4 bytes per
token for its strings and key names (8 bytes for any other value), and 4096 tokens per image,
audio or video item (recognised by its place in the schema, not its content). Such a prompt has
no keys, so nothing is predicted cached for it.
In live runs the prediction matched the engine's reported `cached_tokens` on over 99% of requests.

Hybrid models (attention plus recurrent layers, e.g. Qwen3.8 Flash-Next) need more than the
attention blocks: a hit at position p also needs a saved recurrent state at exactly p. The mirror
tracks each KV group separately and only counts a hit where every group has it. It models full
and MLA attention groups and mamba groups only: a sliding-window, chunked-local or other group
is logged as an error and the mirror then predicts nothing cached, until it next starts over
and does not see that kind again (a group without a kind is taken for full attention). With
speculative decoding (EAGLE, MTP) vLLM drops the last block of a hit; the mirror does the same
only when `speculative_decoding` is set, as nothing in the event stream says so.
Every BlockStored of a hash is taken for a physical copy (a hybrid model re-prefilling from
block 0 stores a second copy of it). vLLM's reuse reports (`kv_cache_report_mode: "full"`) look
the same and would add copies no eviction removes, so the proxy strips that field from requests.
The one report shape the mirror recognises, a parent-less store of several recurrent blocks, is
counted (`reports`) and dropped.

Engines that cache hybrid-model prefixes at a finer unit than the block, such as halt95's vLLM
fork (`--block-size 4096 --prefix-match-unit 64`), save a recurrent state where each prompt ends,
rounded down to the unit, and publish unit-sized block events for it: with speculative decoding,
an attention entry at the prompt's last unit boundary and a checkpoint one unit lower; without
it, both at the last boundary before the prompt's last token. The
next turn resumes there, at for example 2 blocks and 31 units (10,176 tokens), not at the last
full block. The mirror matches these events. It takes the unit from `prefix_match_unit` in
`vllm:cache_config_info` (ignored unless it divides the block size), else from the size of the
unit events (relearned after each start-over), and follows the engine's rule:
the attention layers reach through the contiguous cached full blocks and then the next block's
unit boundaries, probed from the top but never past the prompt's last token (the engine recomputes it, so
a prompt that ends on a unit boundary cannot read the entry it registers there; a longer one
can; the keys a hold keeps for an admitted prompt drop this bound, as they stand for the next,
longer turn); speculative decoding drops one unit rather than a block;
the hit is the newest boundary at or below that with a recurrent checkpoint, which may lie a
block or more lower (a turn whose decode filled the block its prompt ended in leaves its
checkpoint below a block the next turn matches whole). Unit matching only adds boundaries to
the block-level rule, so it never predicts less. The events carry only the last unit of tokens
of each entry, which repeats elsewhere in agent conversations, so the mirror places them through
the prompts it has keyed: it remembers where each prompt ends, under a key chained over all of
its tokens, and counts a unit boundary only at a remembered end whose entry the engine holds.
That the entry is this prompt's is inferred from its last unit of tokens, not verified. The
mirror remembers the 32,768 most recently consulted ends. A turn that did not go through the
proxy leaves an end the mirror cannot place, so the next turn resumes at the last end it did
key, an earlier turn's. On that engine, a mirror without
unit matching under-predicted cached tokens by a median of 21,824 per request (p90 61,248);
an earlier version of the matching had a median and p90 error of 0 under the same load.

The mirror is only as good as the stream. On every (re)subscription it starts empty and rebuilds
from the publisher's replay buffer; until that replay completes the proxy asks it nothing and
predicts nothing cached, and a replay that fails ends the subscription so it is retried. Batches missing from the replay
buffer are skipped and the mirror starts over from the batch after them (they may have removed
blocks); undecodable batches are skipped. Neither fails every retry. Events the mirror cannot
use are counted and dropped, and the first of each kind is logged: blocks on another medium
than the GPU (CPU or disk KV offload), stores without a usable `group_idx`, and block hashes of
an unknown type. Stores whose parent is unknown or that carry no hashes are counted without a
log line. A removal without a usable `group_idx` is applied to every group, as forgetting only
under-reports. A sequence number at or below the last applied one, other than the live
stream repeating just-replayed batches in order within 5 s, means the engine restarted: the mirror starts over, resyncs from the
replay, and its generation moves so admission sees its holds may be gone. The same goes for a
restart seen inside the replay: each later round asks from the last batch already applied, and a
publisher that does not send it back unchanged fails the replay, which resyncs. Holds are
judged once a second against whatever the mirror holds now, while it is healthy: the first
look records how much of a hold is cached, and a hold that later falls below half of that
(evicted, lost to a restart, or older than the replay buffer reaches) is released. A hold the
mirror never saw (the prompt could not be tokenized, or its blocks carry extra hash keys) cannot
be verified: it is released once it is no longer firm, or when the mirror is healthy again
after starting over.

## Admission

**Budget.** The engine's KV pool, read from `vllm:cache_config_info`, less the block the engine
reserves as its null block. Until it is known, nothing is admitted: requests wait. `max_active`
caps concurrency on top.

**What a running request costs.** Its prompt, its output so far or its expected output
(whichever is larger; the expectation is the 75th percentile of the conversation's last 16
outputs, else a default, and never more than the request's output cap),
and a fixed per-request overhead in blocks for state the engine keeps beyond the tokens. A
waiter is priced when it is granted, so a raised overhead reaches the queue too. The charge
updates as output streams: from the stream's own token counts when it reports usage per chunk,
else the configured `tokens_per_chunk` per choice in each chunk (set it to 1 plus the draft
length under speculative decoding). A non-streaming request is charged its whole output cap from admission until it
finishes: its `max_tokens`, else `default_output_tokens`, since nothing raises its charge while
it runs. The expected output is per sequence: a request with `n` choices or a batched
prompt is charged it once per sequence.

**Holds.** When a turn ends, its context is held: counted against the budget so nothing new is
admitted on top of it. It stays counted while the conversation's next request runs, which is
charged only the blocks it does not share with the hold; a higher-priority waiter may still
displace it then. A hold is firm (cannot be displaced by work of the same or lower
priority) for as long as re-prefilling the context would take, scaled up by the requests running,
whose decoding that prefill would slow, and at least 5 s so a quick tool call always comes back warm, and as
long again from when its conversation's next request starts waiting. After that
it is expendable: it stays while nothing needs the room. A hold has the priority of the turn that
left it; a turn that ends without usage (client gone, engine error) keeps the previous hold.

**The hold model.** A session is a conversation of `/v1/chat/completions`: its model, its
caller and its first two messages (a `/v1/completions` request has none and holds nothing). A
session has at most one hold: a size in whole blocks, the mirror keys of the prompt behind it,
a priority, when it was made, the time its last turn finished, and how much of it the mirror
confirmed. A session is forgotten, hold included, after 24 h without traffic; a full session
table (4096) forgets only sessions holding nothing.

- A hold is made or replaced only when a request of the session finishes, at once (also while
  other requests of the session run). Its size is the prompt the engine reported, bounded by the
  prompt admission priced, so it never exceeds the room the finishing request returns. Without
  usage, a request that streamed output stands in its priced prompt, but only to raise the hold;
  one that streamed nothing changes nothing, not even the hold's firmness. Any other finish,
even one that keeps the hold as it was, restarts its firm window. A smaller turn
  replaces the hold only when it arrived after the hold was made and its reported prompt differs
  from the held one by keys (a compacted context); a side call that overlapped the turn behind
  the hold, or a prefix of the held prompt, keeps it.
- Modelled use is every hold plus every running request's charge less the leading blocks its
  prompt shares with its own session's hold, by mirror keys. A prompt without keys shares
  nothing. Nothing else is counted, so within a conversation a block is counted once; blocks
  two conversations share are counted by each.
- A waiter is admitted only if modelled use after it, and after the holds it displaces, is
  within the budget (a request larger than the pool runs alone).
- Displacing a hold removes it for good; nothing recreates it except a later finish of its own
  session, sized as any hold is. A waiter may always displace its own session's hold, firm or
  not (it is that conversation's next turn). Other holds are displaced by higher-priority work,
  or, when idle and no longer firm, by work of any priority while fewer than 4 requests run.
  A firm hold is never displaced by another conversation's work of the same or lower priority.

These are checked over random sequences of arrivals, finishes, streamed output and time
(`TestHoldModelProperties`).

**Order.** Priority first (lower runs first, as in vLLM). Then a waiter whose prompt is at least
half cached, or whose idle conversation holds at least half of it; then small prompts (under
16K tokens); then cold ones; each by arrival. A blocked waiter reserves the room it still needs, so smaller work cannot starve it.
The exception: a waiter that could not run even with every hold it may displace gone (larger
than the pool less the holds of higher priority) reserves nothing, as holding room back for it
protects nothing.

**Eviction.** Work that is not of higher priority displaces an expendable hold only while fewer
than 4 requests run. On a busy engine displacing a context only adds prefill work.

**Priority is a ceiling.** A caller's identity sets the most urgent it may be; a request may only
make it run later (its `priority` body field). Ordering is strict, so a self-promoted caller
could starve everyone else.

Holds do not control what the engine evicts: vLLM evicts its least recently freed cached blocks.
The proxy can only keep enough room free that the engine has no reason to evict a held context.
An engine-side retention API (vLLM RFC #37003) would let holds become real pins.

**Engine restarts.** A readiness probe lists the engine's models every few seconds. While the
engine is not ready, requests wait, as they would for room, instead of failing; nothing is
replayed. A request that fails after it reached the engine returns its error: replaying it would
hide the failure and repeat the work of a turn that may have half run.

## The per-request overhead

Besides prompt and output tokens a running request holds blocks of its own: recurrent states,
speculative-decoding scratch, prefix-cache checkpoints. On the reference hybrid model this was 17
blocks per request on one engine version, 28 on another, and 7-12 on a third. Too low a figure
means the engine runs out of KV space; too high wastes capacity.

The figure is configuration, measured with `cavp -calibrate` on an idle engine. The proxy cannot
learn it continuously: an engine that keeps finished turns pinned for their next turn reports most
of its pool in use while nothing runs, and does not export how much of that is pinned.

The overcommit guard covers a figure that is too low. Its one signal is the engine preempting a
request: that raises a margin added to every running and waiting request's overhead, by 2
blocks at most once a minute, up to 32, and it falls by one step per 15 quiet minutes. The
engine evicting a held context is not a signal: the engine drops cached prefixes for its own
reasons too, and an earlier version that used it, gated on modelled use near the budget, fired
with nothing running because idle holds alone filled the budget. Neither is the engine's waiting queue: it counts every request not yet
scheduled, including ones waiting for the next step's prefill token budget, and is non-zero in
normal operation. Using it raised the margin by 8 blocks in five minutes of normal load.

## Engine findings along the way

- With `mamba_cache_mode=align`, recurrent checkpoints were written only where a prefill step
  ended on a block boundary (3,200-4,096 tokens), so a turn adding less than a block missed
  entirely. A finer checkpoint unit fixed it (`--prefix-match-unit`, in halt95's fork); the
  mirror matches its unit-sized entries (above).
- On one engine version a prompt's tail checkpoint was released early and aged out of the cache
  before the agent returned (fixed in halt95's vLLM fork, v2.5.1).
- Each was found by tracing KV events against real misses, not by aggregate hit rates.

## Results

10 coding agents for 6.4 hours on 4x RTX 3090 (Qwen3.8 Flash-Next, 262K context): 91% of prompt
tokens served from cache, 95% of quick returns cached, zero preemptions, against 13-16% with
slot-based admission at the start.
