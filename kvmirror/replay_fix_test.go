package kvmirror

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

type rawBatch struct {
	seq     int64
	payload []byte
}

func batchPayload(evs ...Event) []byte {
	b, _ := msgpack.Marshal([]any{1.0, evs, 0})
	return b
}

// serveReplay answers every replay request with the batches from the requested sequence
// on, then the end marker. It returns the endpoint.
func serveReplay(ctx context.Context, t *testing.T, buf []rawBatch) string {
	t.Helper()
	router := zmq4.NewRouter(ctx)
	t.Cleanup(func() { router.Close() })
	if err := router.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	seqb := func(v int64) []byte {
		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, uint64(v))
		return b
	}
	go func() {
		for {
			msg, err := router.Recv()
			if err != nil || len(msg.Frames) < 3 {
				return
			}
			id, from := msg.Frames[0], int64(binary.BigEndian.Uint64(msg.Frames[2]))
			for _, b := range buf {
				if b.seq >= from {
					router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte("kv"), seqb(b.seq), b.payload))
				}
			}
			router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte("kv"), seqb(-1), []byte{}))
		}
	}()
	return "tcp://" + router.Addr().String()
}

// A batch lost for good inside the publisher's replay buffer, or one that cannot be
// decoded, is skipped: the replay completes with what is there instead of failing every
// resubscribe (review r4-85 #1).
func TestFixReplaySkipsHoleAndUndecodable(t *testing.T) {
	old := replayFirstWait
	replayFirstWait = 300 * time.Millisecond
	defer func() { replayFirstWait = old }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	toks := seqPrompt(13)
	for name, buf := range map[string][]rawBatch{
		"hole": {
			{1, batchPayload(attnStore(1, nil, toks[:4]))},
			{2, batchPayload(attnStore(2, uint64(1), toks[4:8]))},
			{4, batchPayload(attnStore(3, uint64(2), toks[8:12]))},
		},
		"undecodable": {
			{1, batchPayload(attnStore(1, nil, toks[:4]))},
			{2, batchPayload(attnStore(2, uint64(1), toks[4:8]))},
			{3, []byte{0xc1, 0xc1}},
			{4, batchPayload(attnStore(3, uint64(2), toks[8:12]))},
		},
	} {
		m := New()
		if _, err := m.replay(ctx, serveReplay(ctx, t, buf)); err != nil {
			t.Fatalf("%s: replay failed: %v", name, err)
		}
		if got := m.CachedTokens(toks); got != 12 {
			t.Fatalf("%s: cached %d after replay, want 12", name, got)
		}
	}
}
