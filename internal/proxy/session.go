package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
)

// sessionKey fingerprints a chat conversation so successive turns of the same
// agent or chat map to the same admission session. The system prompt and the
// first user message are replayed verbatim on every turn of a conversation,
// while later messages grow, so the first two messages are the stable prefix.
// Distinct conversations that share both (same system prompt and identical
// opening message, e.g. several chat users greeting the same assistant)
// collapse into one session. They share cached blocks only up to the common
// prefix; their divergent suffixes are then undercounted by the budget, which
// degrades toward pre-budget behavior for exactly those conversations. Clients
// that can name their conversation should send X-Session-Key instead, which
// sessionKeyFromID turns into a collision-free key.
//
// Returns empty (ephemeral, no budget tracking) when there are no messages.
func sessionKey(target, owner string, body map[string]any) string {
	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) == 0 {
		return ""
	}
	n := min(2, len(msgs))
	h := sha256.New()
	io.WriteString(h, target)
	// The caller's identity keeps two users with the same opening apart: sessions now
	// carry admission state (cache holds), not just log correlation.
	io.WriteString(h, "\x00"+owner+"\x00")
	enc := json.NewEncoder(h) // deterministic: encoding/json sorts map keys
	for _, m := range msgs[:n] {
		if err := enc.Encode(m); err != nil {
			return ""
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// sessionKeyFromID keys a session on a caller-chosen identifier (the
// X-Session-Key header). Hashed with the target so the raw identifier never
// appears in logs and the same conversation against two models stays distinct,
// and namespaced by the authenticated identity so one caller cannot merge its
// accounting into another's by guessing their identifier.
func sessionKeyFromID(target, owner, id string) string {
	h := sha256.New()
	io.WriteString(h, target)
	io.WriteString(h, "\x00")
	io.WriteString(h, owner)
	io.WriteString(h, "\x00")
	io.WriteString(h, id)
	return hex.EncodeToString(h.Sum(nil))[:16]
}
