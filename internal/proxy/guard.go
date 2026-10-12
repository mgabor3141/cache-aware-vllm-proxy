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
// blocks pinned. So the configured figure is the source of truth (measure it with
// `cavp -calibrate`), and this guard is a safety net for when it is too low.
//
// Its one signal is the engine preempting a request (vllm:num_preemptions_total rising): the
// proxy let in more than the engine could hold. Each raises a margin added to every running
// and waiting request's overhead, by guardStepBlocks, at most once per guardCooldown and up to
// guardMaxMarginBlocks. Each guardDecayAfter without a signal the margin falls by one step,
// back to zero (the configured figure). Every change is logged.
//
// Not signals: the engine evicting a held context (it also drops cached prefixes for its own
// reasons, and a hold is only a reservation), and the engine's waiting queue, which counts
// every request not yet scheduled, including one admitted a moment ago and one waiting behind
// another's prefill for the step's token budget.
//
// The guard cannot tell overhead from other causes: a request that outruns its expected
// output, or traffic that reaches the engine without passing through the proxy, look the
// same. It treats them all as margin.
const (
	guardStepBlocks      = 2
	guardMaxMarginBlocks = 32
	guardCooldown        = time.Minute
	guardDecayAfter      = 15 * time.Minute
)

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

// EngineSignals reports the engine's preemption counters, one per metric series (its
// label set; a data-parallel engine has one per engine). A series counts when it rises
// from its last value; a new series, or one that went down (the engine restarted), only
// sets its baseline; a series missing from one read keeps it. A read without any series
// means the guard is blind: that is logged once until a series appears, which then
// starts from a new baseline.
func (q *AdmissionQueue) EngineSignals(preemptions map[string]float64) {
	q.mu.Lock()
	defer q.mu.Unlock()
	rose := false
	for series, v := range preemptions {
		if last, ok := q.lastPreempt[series]; ok && v > last {
			rose = true
		}
	}
	if len(preemptions) == 0 {
		q.lastPreempt = nil
		if !q.preemptBlind {
			q.preemptBlind = true
			log.Printf("level=warn msg=\"engine metrics have no vllm:num_preemptions_total: the overcommit guard cannot see preemptions\"")
		}
		return
	}
	q.preemptBlind = false
	if q.lastPreempt == nil {
		q.lastPreempt = map[string]float64{}
	}
	for series, v := range preemptions { // a series missing from one read keeps its baseline
		q.lastPreempt[series] = v
	}
	if rose {
		q.guardRaiseLocked("engine preempted", q.now())
	}
}
