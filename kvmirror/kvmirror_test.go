package kvmirror

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
)

type fixtureRequest struct {
	Tokens []int64 `json:"tokens"`
	Cached int     `json:"cached"`
}

func loadFixture(t *testing.T) ([][]byte, []fixtureRequest) {
	t.Helper()
	files, _ := filepath.Glob("testdata/kvevents/*.bin")
	sort.Strings(files)
	var batches [][]byte
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		batches = append(batches, b)
	}
	raw, err := os.ReadFile("testdata/kvevents/requests.json")
	if err != nil {
		t.Fatal(err)
	}
	var reqs []fixtureRequest
	if err := json.Unmarshal(raw, &reqs); err != nil {
		t.Fatal(err)
	}
	return batches, reqs
}

// Real events captured from a hybrid Qwen3.8 Flash-Next engine (testdata/kvevents, 2026-10-07): request 0 is a fresh
// 19,557-token conversation, request 1 its second turn, request 2 a new conversation
// sharing the system prompt. The engine reported 0, 16,000 and 16,000 cached tokens.
func TestKVMirrorReplaysCapturedEvents(t *testing.T) {
	batches, reqs := loadFixture(t)
	m := New()
	m.SetSpeculative(true)
	var after [][]int
	for _, b := range batches {
		seq := int64(binary.BigEndian.Uint64(b[:8]))
		events, err := decodeBatch(b[8:])
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		m.Apply(seq, events)
		var row []int
		for _, r := range reqs {
			row = append(row, m.CachedTokens(r.Tokens))
		}
		after = append(after, row)
	}
	st := m.Status()
	t.Logf("status %+v", st)
	for i, row := range after {
		t.Logf("after batch %2d: predicted %v", i, row)
	}
	if st.BlockSize != 3200 || st.Stats.Stored == 0 {
		t.Fatalf("mirror did not learn the cache: %+v", st)
	}
	// Requests 1 and 2 hit exactly 16,000 tokens in the engine.
	for i := 1; i <= 2; i++ {
		if got := m.CachedTokens(reqs[i].Tokens); got != reqs[i].Cached {
			t.Fatalf("request %d: mirror %d, engine %d", i, got, reqs[i].Cached)
		}
	}
	// A shorter branch of request 0 (5 blocks and one token): the engine would match 5
	// attention blocks, drop one, and find no checkpoint at or below 4.
	if got := m.CachedTokens(reqs[0].Tokens[:5*3200+1]); got != 0 {
		t.Fatalf("shorter branch: mirror %d, want 0", got)
	}
}

func TestKVMirrorRestartClears(t *testing.T) {
	m := New()
	m.SetSpeculative(true)
	g := 0
	toks := make([]int64, 9)
	m.Apply(5, []Event{{Type: "BlockStored", BlockHashes: []any{uint64(1), uint64(2)}, TokenIDs: toks[:8], BlockSize: 4, GroupIdx: &g, Kind: "full_attention"}})
	if m.CachedTokens(toks) != 4 {
		t.Fatal("blocks not stored")
	}
	m.Apply(5000, nil)
	m.Apply(0, nil) // sequence went far backwards: engine restarted
	if m.CachedTokens(toks) != 0 {
		t.Fatal("mirror kept blocks across an engine restart")
	}
}

func TestKVMirrorHybridRule(t *testing.T) {
	m := New()
	m.SetSpeculative(true)
	attn, mamba := 5, 0
	toks := make([]int64, 13)
	for i := range toks {
		toks[i] = int64(i)
	}
	store := func(g *int, kind string, h uint64, parent any, from int) {
		m.Apply(-1, []Event{{Type: "BlockStored", BlockHashes: []any{h}, ParentBlockHash: parent, TokenIDs: toks[from : from+4], BlockSize: 4, GroupIdx: g, Kind: kind}})
	}
	// Attention 1..3 contiguous; the speculative-decoding drop leaves 2; the hit is the
	// newest block at or below 2 with a mamba checkpoint.
	store(&attn, "full_attention", 1, nil, 0)
	store(&attn, "full_attention", 2, uint64(1), 4)
	store(&attn, "full_attention", 3, uint64(2), 8)
	store(&mamba, "mamba", 1, nil, 0)
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("mamba checkpoint only at block 1: got %d, want 4", got)
	}
	store(&mamba, "mamba", 3, uint64(2), 8)
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("checkpoint at 3 is past the dropped block: got %d, want 4", got)
	}
	store(&mamba, "mamba", 2, uint64(1), 4)
	if got := m.CachedTokens(toks); got != 8 {
		t.Fatalf("checkpoint at 2: got %d, want 8", got)
	}
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(3)}, GroupIdx: &attn}})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("tail attention block evicted: got %d, want 4", got)
	}
}

// vLLM can cache two physical copies of one hash (a hybrid model recomputes the attention
// blocks past its last recurrent checkpoint); evicting one publishes BlockRemoved while the
// other stays. The mirror must keep the block until every copy is gone.
func TestKVMirrorDuplicateCopies(t *testing.T) {
	m := New()
	g := 5
	toks := make([]int64, 9)
	toks[0] = 1
	first := Event{Type: "BlockStored", BlockHashes: []any{uint64(6)}, TokenIDs: toks[:4], BlockSize: 4, GroupIdx: &g, Kind: "full_attention"}
	st := Event{Type: "BlockStored", BlockHashes: []any{uint64(7)}, ParentBlockHash: uint64(6), TokenIDs: toks[4:8], BlockSize: 4, GroupIdx: &g, Kind: "full_attention"}
	m.Apply(-1, []Event{first, st, st})
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(7)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 8 {
		t.Fatalf("one of two copies evicted: got %d, want 8", got)
	}
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(7)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("both copies evicted: got %d, want 4", got)
	}
}

func TestReplayEndpointIsNextPort(t *testing.T) {
	if got := ReplayEndpoint("tcp://10.0.0.2:5557"); got != "tcp://10.0.0.2:5558" {
		t.Fatalf("got %q", got)
	}
	if got := ReplayEndpoint("tcp://host"); got != "" {
		t.Fatalf("no port: got %q", got)
	}
}

func attnStore(hash uint64, parent any, toks []int64) Event {
	g := 5
	return Event{Type: "BlockStored", BlockHashes: []any{hash}, ParentBlockHash: parent, TokenIDs: toks, BlockSize: 4, GroupIdx: &g, Kind: "full_attention"}
}

func seqPrompt(n int) []int64 {
	toks := make([]int64, n)
	for i := range toks {
		toks[i] = int64(i + 1)
	}
	return toks
}

// An engine restart resets the publisher's sequence. Without a replay in progress, any
// sequence that goes backwards is a restart, however small the step (review cv-52 #1).
func TestKVMirrorRestartInsideFormerDuplicateWindow(t *testing.T) {
	m := New()
	m.SetSpeculative(true)
	toks := seqPrompt(9)
	m.Apply(0, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	m.Apply(1, nil)
	m.Apply(2, nil)
	if m.CachedTokens(toks) != 4 {
		t.Fatal("blocks not stored")
	}
	m.Apply(0, nil) // the restarted engine's first batch
	if got := m.CachedTokens(toks); got != 0 {
		t.Fatalf("mirror kept a dead engine's blocks: %d cached", got)
	}
	if st := m.Status(); st.Stats.Restarts != 1 {
		t.Fatalf("restart not counted: %+v", st.Stats)
	}
	// The new engine's batches are applied, not dropped as duplicates.
	m.Apply(1, []Event{attnStore(7, nil, toks[:4]), attnStore(8, uint64(7), toks[4:8])})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("new engine's blocks: %d cached, want 4", got)
	}
}

// After a replay, live delivery repeats the replayed batches in order: those are
// skipped, however many there are (a deep backlog must not look like a restart), and the
// stream then continues. A repeat that goes backwards is a restart.
func TestKVMirrorReplayOverlap(t *testing.T) {
	m := New()
	m.SetSpeculative(true)
	toks := seqPrompt(9)
	m.Apply(14000, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	for seq := int64(14001); seq <= 20000; seq++ { // the replay
		m.Apply(seq, nil)
	}
	m.endReplayLocked() // as subscribe does when the replay completes
	for _, seq := range []int64{14000, 14001, 20000} {
		if !m.Apply(seq, []Event{{Type: "AllBlocksCleared"}}) {
			t.Fatalf("seq %d: reported a gap", seq)
		}
	}
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("a replayed batch was applied again: %d cached", got)
	}
	if !m.Apply(20001, nil) {
		t.Fatal("the next live batch is a gap")
	}
	m.Apply(20000, nil)
	if st := m.Status(); st.Stats.Restarts != 1 || st.Stats.Duplicates != 3 {
		t.Fatalf("after the overlap a repeat is a restart: %+v", st.Stats)
	}

	m = New()
	m.Apply(30, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	for seq := int64(31); seq <= 50; seq++ {
		m.Apply(seq, nil)
	}
	m.endReplayLocked()
	m.Apply(40, nil)
	m.Apply(0, nil) // inside the overlap, but behind the previous live batch
	if st := m.Status(); st.Stats.Restarts != 1 {
		t.Fatalf("a restart during the overlap went unnoticed: %+v", st.Stats)
	}
}

// One event whose parent the mirror never saw is skipped; the batch's other events are
// applied (review cv-52 #14). A block hash the mirror cannot read still chains the
// blocks after it.
func TestKVMirrorOrphanEventSkipsOnlyItself(t *testing.T) {
	m := New()
	toks := seqPrompt(13)
	g := 5
	m.Apply(-1, []Event{
		attnStore(9, uint64(12345), seqPrompt(4)), // orphan
		{Type: "BlockStored", BlockHashes: []any{"?", uint64(2), uint64(3)}, TokenIDs: toks[:12], BlockSize: 4, GroupIdx: &g, Kind: "full_attention"},
	})
	if st := m.Status(); st.Stats.SkippedOrphan != 1 || st.Blocks != 2 {
		t.Fatalf("status %+v", st)
	}
	// Block 1 has no engine hash to find it by; blocks 2 and 3 are cached but not a prefix.
	if got := m.CachedTokens(toks); got != 0 {
		t.Fatalf("cached %d without the first block", got)
	}
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4])})
	if got := m.CachedTokens(toks); got != 12 {
		t.Fatalf("cached %d, want 12", got)
	}
}

// Keys carry the block size they were built at; under another geometry they match
// nothing rather than being rescaled (review cv-52 #2).
func TestKVMirrorKeysAtAnotherBlockSizeMatchNothing(t *testing.T) {
	m := New()
	toks := seqPrompt(9)
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	keys := m.NewPromptKeys(toks)
	if got := m.Cached(keys); got != 8 {
		t.Fatalf("cached %d, want 8", got)
	}
	m.SetGeometry(8, 0)
	if got := m.Cached(keys); got != 0 {
		t.Fatalf("keys built at 4 read %d under block size 8", got)
	}
}

// Two engine hashes for one block: evicting one and storing the other again keeps the
// block findable (review cv-52 #5).
func TestKVMirrorAliasRestoredAfterEviction(t *testing.T) {
	m := New()
	toks := seqPrompt(5)
	g := 5
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4]), attnStore(2, nil, toks[:4])})
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(1), uint64(2)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 0 {
		t.Fatalf("evicted block cached %d", got)
	}
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4])})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("re-stored block cached %d, want 4", got)
	}
}

// A subscription is healthy only once its replay completed: a replay that fails leaves
// the mirror unhealthy and the subscription retries (review cv-52 #6). Once the replay
// reaches the end marker, the mirror holds the replayed blocks and is healthy.
func TestKVMirrorHealthyOnlyAfterReplay(t *testing.T) {
	old := replayFirstWait
	replayFirstWait = 300 * time.Millisecond
	defer func() { replayFirstWait = old }()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pub := zmq4.NewPub(ctx)
	defer pub.Close()
	if err := pub.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	pubEP := "tcp://" + pub.Addr().String()

	// A replay service that never answers.
	silent := zmq4.NewRouter(ctx)
	defer silent.Close()
	if err := silent.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	m := New()
	m.SetSpeculative(true)
	rctx, rcancel := context.WithCancel(ctx)
	go m.Run(rctx, pubEP, "tcp://"+silent.Addr().String())
	time.Sleep(1500 * time.Millisecond)
	if m.Healthy() {
		t.Fatal("healthy after a failed replay")
	}
	rcancel()

	toks := seqPrompt(9)
	payload, _ := msgpack.Marshal([]any{1.0, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])}, 0})
	router := zmq4.NewRouter(ctx)
	defer router.Close()
	if err := router.Listen("tcp://127.0.0.1:0"); err != nil {
		t.Skipf("zmq listen: %v", err)
	}
	go func() {
		msg, err := router.Recv()
		if err != nil {
			return
		}
		id := msg.Frames[0]
		seq := make([]byte, 8)
		binary.BigEndian.PutUint64(seq, 7)
		router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte("kv"), seq, payload))
		router.Send(zmq4.NewMsgFrom(id, []byte{}, []byte{}, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}, []byte{}))
	}()
	m2 := New()
	m2.SetSpeculative(true)
	go m2.Run(ctx, pubEP, "tcp://"+router.Addr().String())
	deadline := time.Now().Add(5 * time.Second)
	for !m2.Healthy() && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !m2.Healthy() {
		t.Fatal("not healthy after a complete replay")
	}
	if got := m2.CachedTokens(toks); got != 4 {
		t.Fatalf("replayed blocks: %d cached, want 4", got)
	}
}

// A restarted engine's sequence can fall inside the range the last replay covered. A
// batch below the replay's first sequence cannot be a repeat, and once the overlap
// window has passed (an idle engine that restarts later) none can: both are restarts,
// and the new engine's blocks are applied (review 3, rev-2 #1).
func TestKVMirrorRestartInsideReplayedRange(t *testing.T) {
	old, fresh := seqPrompt(9), seqPrompt(9)
	for i := range fresh {
		fresh[i] += 1000
	}
	replay := func(from int64) *Mirror {
		m := New()
		m.SetSpeculative(true)
		m.Apply(from, []Event{attnStore(1, nil, old[:4]), attnStore(2, uint64(1), old[4:8])})
		for seq := from + 1; seq <= 10; seq++ {
			m.Apply(seq, nil)
		}
		m.endReplayLocked()
		return m
	}
	check := func(m *Mirror) {
		t.Helper()
		if o, n := m.CachedTokens(old), m.CachedTokens(fresh); o != 0 || n != 4 || m.Status().Stats.Restarts != 1 {
			t.Fatalf("old cached %d, new cached %d, %+v", o, n, m.Status().Stats)
		}
	}

	m := replay(0)
	m.replayedAt = time.Now().Add(-overlapWindow) // the engine sat idle, then restarted
	m.Apply(0, []Event{attnStore(7, nil, fresh[:4]), attnStore(8, uint64(7), fresh[4:8])})
	check(m)

	m = replay(5)
	m.Apply(2, []Event{attnStore(7, nil, fresh[:4]), attnStore(8, uint64(7), fresh[4:8])})
	check(m)
}

// Blocks stored at another size than the engine's block size and unit (metrics and events
// disagreeing) can never match a prompt's keys; that is logged once instead of silently
// reading 0 (review 3, rev-2 #5). Unit-sized blocks are not a mismatch
// (TestKVMirrorUnitHitDeployedShape).
func TestKVMirrorWarnsOnBlockSizeMismatch(t *testing.T) {
	var logs bytes.Buffer
	log.SetOutput(&logs)
	defer log.SetOutput(os.Stderr)
	m := New()
	m.SetSpeculative(true)
	m.SetGeometry(8, 2) // 4 is neither the block size nor the unit
	toks := seqPrompt(9)
	m.Apply(-1, []Event{attnStore(1, nil, toks[:4]), attnStore(2, uint64(1), toks[4:8])})
	if n := strings.Count(logs.String(), "block size differs"); n != 1 {
		t.Fatalf("logged %d times, want once:\n%s", n, logs.String())
	}
}
