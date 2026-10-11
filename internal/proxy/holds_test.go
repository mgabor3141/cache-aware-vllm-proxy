package proxy

import (
	"testing"
)

// A session's held prefix stays in the budget while one of its own requests runs: the
// engine still has it cached, so the room is not free for others (invariants #1).
func TestHoldChargedWhileOwnRequestRuns(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	ta := q.start("a", 0, 300000)
	ta.finish(mustAdmit(t, ta, "a's turn"), 300000, 0)
	idle := q.metrics().usedTokens
	if idle != 300800 {
		t.Fatalf("setup: used %d, want the 300800-token hold", idle)
	}
	side := q.start("a", 0, 2000)
	rs := mustAdmit(t, side, "a's side call")
	if used := q.metrics().usedTokens; used < idle {
		t.Errorf("while a's side call runs, modelled use fell %d -> %d: the held prefix left the budget", idle, used)
	}
	x := q.start("x", 0, 400000) // fits only if a's hold is forgotten
	mustWait(t, x, "x beside a's running side call and its hold")
	rs()
	q.tick()
}
