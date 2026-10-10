package proxy

import (
	"log"
	"time"
)

// Overcommit guard.
//
// Admission charges every running request a fixed overhead in blocks (prefill_overhead_blocks,
// decode_overhead_blocks): recurrent state, speculative and checkpoint blocks the engine
// allocates beyond the prompt and output. The right figure depends on the model and the engine
// version, and the engine's own usage cannot measure it once the engine keeps finished turns'
// blocks pinned (grace pins in halt95's vLLM fork, v2.5 hold most of the pool while nothing runs). So the
// configured figure is the source of truth (measure it with `cavp -calibrate`), and this guard
// is a safety net for when it is too low.
//
// Two signals mean the proxy let in more than the engine could hold:
//
//	early  the engine evicted a hold the proxy still counted as firm (seen through KV events)
//	       while the proxy's own accounting was near the budget. An eviction with plenty of
//	       modelled room is not counted: the engine also drops cached prefixes for its own
//	       reasons (halt95's vLLM fork, v2.5, pins at most 16 finished turns), and a low overhead would show
//	       up near the budget first.
//	late   the engine preempted a request (num_preemptions_total).
//
// Each raises a margin added to every running request's overhead, by guardStepBlocks, at most
// once per guardCooldown and up to guardMaxMarginBlocks. After guardDecayAfter without a
// signal the margin falls by one step, back to zero (the configured figure).
//
// Not a signal: the engine's waiting queue (num_requests_waiting, or its "capacity" reason).
// It counts every request not yet scheduled, including one admitted a moment ago and one
// waiting behind another's prefill for the step's token budget, so it is non-zero in normal
// operation and says nothing about KV space.
//
// The guard cannot tell overhead from other causes: a request that outruns its expected
// output, or traffic that reaches the engine without passing through the proxy, look the
// same. It treats them all as margin, and logs each adjustment with its reason.
const (
	guardStepBlocks      = 2
	guardMaxMarginBlocks = 32
	guardCooldown        = time.Minute
	guardDecayAfter      = 15 * time.Minute
	// guardPressure: the early signal counts only when modelled use is at least this share
	// of the budget.
	guardPressure = 0.85
)

// guardEvictionLocked is the early signal: the engine evicted a hold the proxy counted as firm.
// evicted is the hold's size, which the accounting still includes.
func (q *AdmissionQueue) guardEvictionLocked(now time.Time, evicted int) {
	if b := q.budget(); b > 0 && float64(q.cacheUsedLocked(now)-evicted) >= guardPressure*float64(b) {
		q.guardRaiseLocked("engine evicted a firm hold near the budget", now)
	}
}

// overheadLocked is the per-request overhead in blocks, including the guard's margin.
func (q *AdmissionQueue) overheadLocked(decoding bool) int {
	return q.cfg.Cache.overhead(decoding) + q.margin
}

func (q *AdmissionQueue) guardRaiseLocked(reason string, now time.Time) {
	if !q.lastRaise.IsZero() && now.Sub(q.lastRaise) < guardCooldown {
		q.lastSignal = now
		return
	}
	q.lastRaise, q.lastSignal = now, now
	if q.margin >= guardMaxMarginBlocks {
		log.Printf("level=warn msg=\"overcommit guard: margin at its cap\" reason=%q margin_blocks=%d active=%d", reason, q.margin, q.active)
		return
	}
	q.margin = min(q.margin+guardStepBlocks, guardMaxMarginBlocks)
	q.marginRaises++
	log.Printf("level=warn msg=\"overcommit guard raised the margin\" reason=%q margin_blocks=%d prefill_overhead=%d decode_overhead=%d active=%d",
		reason, q.margin, q.overheadLocked(false), q.overheadLocked(true), q.active)
}

func (q *AdmissionQueue) guardDecayLocked(now time.Time) {
	if q.margin == 0 || now.Sub(q.lastSignal) < guardDecayAfter {
		return
	}
	q.margin = max(q.margin-guardStepBlocks, 0)
	q.lastSignal = now
	log.Printf("level=info msg=\"overcommit guard lowered the margin\" margin_blocks=%d", q.margin)
}

// EngineSignals reports the engine's preemption counter.
func (q *AdmissionQueue) EngineSignals(preemptions float64) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.lastPreempt >= 0 && preemptions > q.lastPreempt {
		q.guardRaiseLocked("engine preempted", q.now())
	}
	q.lastPreempt = preemptions
}
