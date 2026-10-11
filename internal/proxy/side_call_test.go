package proxy

import "testing"

func seqTok(start, n int) []int64 {
	p := make([]int64, n)
	for i := range p {
		p[i] = int64(start + i)
	}
	return p
}

// A keyed side call that overlapped the main turn and finishes after it keeps the main
// turn's hold (fixrev2 rev-1 #1).
func TestKeyedSideCallKeepsMainHold(t *testing.T) {
	q, _ := newCacheQueue(t, nil)
	main := q.startReq("s", 0, 242000, seqTok(1, 242000))
	rm := mustAdmit(t, main, "main")
	side := q.startReq("s", 0, 3500, seqTok(10_000_000, 3500))
	rs := mustAdmit(t, side, "side")
	main.finish(rm, 242000, 4000)
	side.finish(rs, 3500, 50)
	if h := q.sessionHeld("s"); h < 240000 {
		t.Fatalf("hold %d, want the main turn's", h)
	}
}

// A compacted next turn (arrives after the hold, smaller, different keys) replaces it.
func TestCompactedTurnReplacesHold(t *testing.T) {
	q, clk := newCacheQueue(t, nil)
	main := q.startReq("s", 0, 242000, seqTok(1, 242000))
	main.finish(mustAdmit(t, main, "main"), 242000, 4000)
	clk.add(1)
	t3 := q.startReq("s", 0, 20000, append(seqTok(1, 3200), seqTok(50_000_000, 16800)...))
	r3 := mustAdmit(t, t3, "compacted")
	clk.add(1)
	t3.finish(r3, 20000, 10)
	if h := q.sessionHeld("s"); h > 30000 {
		t.Fatalf("hold %d, want the compacted turn's", h)
	}
}
