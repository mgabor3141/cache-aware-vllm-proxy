package kvmirror

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
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

// vLLM can cache two physical copies of one hash; evicting one publishes BlockRemoved
// while the other stays. The mirror must keep the block until every copy is gone.
func TestKVMirrorDuplicateCopies(t *testing.T) {
	m := New()
	m.dropLast = false
	g := 5
	toks := make([]int64, 5)
	st := Event{Type: "BlockStored", BlockHashes: []any{uint64(7)}, TokenIDs: toks[:4], BlockSize: 4, GroupIdx: &g, Kind: "full_attention"}
	m.Apply(-1, []Event{st, st})
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(7)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 4 {
		t.Fatalf("one of two copies evicted: got %d, want 4", got)
	}
	m.Apply(-1, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(7)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 0 {
		t.Fatalf("both copies evicted: got %d, want 0", got)
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

// The event sequence captured from the engine with --prefix-match-unit 64 (2026-10-08):
// prompt 7,987 tokens; full blocks 0 and 1; a tail checkpoint ending at 7,872 in every
// recurrent group (parent: an unannounced span); and the attention span [7872, 7936)
// whose parent is that checkpoint. The next turn (8,483 tokens) was served 7,872.
func TestKVMirrorFineGrainedTailHit(t *testing.T) {
	m := New()
	m.SetGeometry(3200, 64)
	toks := make([]int64, 8483)
	for i := range toks {
		toks[i] = int64(i*7919%50000 + 1)
	}
	ev := func(g int, kind string, bs int, hash, parent uint64, from int) Event {
		e := Event{Type: "BlockStored", BlockHashes: []any{hash}, TokenIDs: toks[from : from+bs], BlockSize: bs, GroupIdx: &g, Kind: kind}
		if parent != 0 {
			e.ParentBlockHash = parent
		}
		return e
	}
	var evs []Event
	for g := 0; g < 4; g++ {
		evs = append(evs, ev(g, "mamba", 3200, 11, 0, 0))
	}
	evs = append(evs, ev(5, "mla_attention", 3200, 11, 0, 0), ev(5, "mla_attention", 3200, 22, 11, 3200))
	for g := 0; g < 4; g++ {
		evs = append(evs, ev(g, "mamba", 64, 33, 999, 7808))
	}
	evs = append(evs, ev(5, "mla_attention", 64, 44, 33, 7872))
	m.Apply(1, evs)
	if got := m.CachedTokens(toks); got != 7872 {
		t.Fatalf("next turn cached %d, want 7872", got)
	}
	if got := m.CachedTokens(toks[:7930]); got != 3200 {
		t.Fatalf("a prompt ending inside the attention span cached %d, want the block-level 3200", got)
	}
	g := 0
	m.Apply(2, []Event{{Type: "BlockRemoved", BlockHashes: []any{uint64(33)}, GroupIdx: &g}})
	if got := m.CachedTokens(toks); got != 3200 {
		t.Fatalf("after losing one recurrent checkpoint cached %d, want 3200", got)
	}
}
