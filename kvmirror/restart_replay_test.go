package kvmirror

import (
	"context"
	"encoding/binary"
	"log"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
)

// twoLivesReplay answers the first replay request from old without the end marker (its
// tail lost at the ROUTER queue), and later ones like a restarted engine: its own buffer
// filtered by seq >= start, then the marker (vLLM's _service_replay).
func twoLivesReplay(ctx context.Context, t *testing.T, old, fresh []rawBatch) string {
	t.Helper()
	router := zmq4.NewRouter(ctx)
	t.Cleanup(func() { router.Close() })
	if err := router.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	se := func(v int64) []byte {
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, uint64(v))
		return b
	}
	go func() {
		for n := 1; ; n++ {
			msg, err := router.Recv()
			if err != nil || len(msg.Frames) < 3 {
				return
			}
			id, from := msg.Frames[0], int64(binary.BigEndian.Uint64(msg.Frames[2]))
			buf := old
			if n > 1 {
				buf = fresh
			}
			for _, b := range buf {
				if b.seq >= from || n == 1 {
					router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte("kv"), se(b.seq), b.payload))
				}
			}
			if n > 1 {
				router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte("kv"), se(-1), []byte{}))
			}
		}
	}()
	return "tcp://" + router.Addr().String()
}

// An engine restart between two replay rounds: the new engine's buffer holds nothing at
// or above the sequence asked for, so the round is the marker alone. The mirror must not
// go healthy over the dead engine's blocks; it resyncs and rebuilds from the new engine.
func TestReplayRestartBetweenRoundsResyncs(t *testing.T) {
	oldWait := replayFirstWait
	replayFirstWait = 250 * time.Millisecond
	defer func() { replayFirstWait = oldWait }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	oldToks, newToks := seqPrompt(9), seqPrompt(9)
	for i := range newToks {
		newToks[i] += 61000
	}
	oldBuf := []rawBatch{
		{0, batchPayload(attnStore(1, nil, oldToks[:4]), attnStore(2, uint64(1), oldToks[4:8]))},
		{1, batchPayload(Event{Type: "BlockRemoved", BlockHashes: []any{uint64(501)}, GroupIdx: ptrInt(5)})},
		{2, batchPayload(Event{Type: "BlockRemoved", BlockHashes: []any{uint64(502)}, GroupIdx: ptrInt(5)})},
	}
	newBuf := []rawBatch{{0, batchPayload(attnStore(7, nil, newToks[:4]), attnStore(8, uint64(7), newToks[4:8]))}}
	replay := twoLivesReplay(ctx, t, oldBuf, newBuf)

	pub := zmq4.NewPub(ctx)
	t.Cleanup(func() { pub.Close() })
	if err := pub.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	m := New()
	go m.Run(ctx, "tcp://"+pub.Addr().String(), replay)
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if m.Healthy() && m.CachedTokens(newToks) > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !m.Healthy() || m.CachedTokens(oldToks) != 0 || m.CachedTokens(newToks) == 0 {
		t.Fatalf("healthy=%v dead engine's prompt %d cached, live engine's %d: want 0 and >0",
			m.Healthy(), m.CachedTokens(oldToks), m.CachedTokens(newToks))
	}
}

func ptrInt(v int) *int { return &v }

// The production shape: a round that takes the buffer's tail loses the end marker (the
// ROUTER's queue limit), so the next round, asked from the batch after the tail, gets the
// marker alone. That is the same engine, and the replay completes.
func TestReplayTailWithoutMarkerCompletes(t *testing.T) {
	oldWait := replayFirstWait
	replayFirstWait = 250 * time.Millisecond
	defer func() { replayFirstWait = oldWait }()
	logs := &syncBuf{}
	log.SetOutput(logs)
	defer log.SetOutput(os.Stderr)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	toks := seqPrompt(9)
	buf := []rawBatch{
		{0, batchPayload(attnStore(1, nil, toks[:4]))},
		{1, batchPayload(attnStore(2, uint64(1), toks[4:8]))},
	}
	replay := twoLivesReplay(ctx, t, buf, buf)
	pub := zmq4.NewPub(ctx)
	t.Cleanup(func() { pub.Close() })
	if err := pub.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	m := New()
	go m.Run(ctx, "tcp://"+pub.Addr().String(), replay)
	waitHealthy := time.Now().Add(10 * time.Second)
	for !m.Healthy() && time.Now().Before(waitHealthy) {
		time.Sleep(20 * time.Millisecond)
	}
	if !m.Healthy() || m.CachedTokens(toks) != 8 || strings.Contains(logs.String(), "resyncing") {
		t.Fatalf("healthy=%v cached=%d logs=%s", m.Healthy(), m.CachedTokens(toks), logs.String())
	}
}

type syncBuf struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuf) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}
