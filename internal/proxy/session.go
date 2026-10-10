package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
)

// sessionKey fingerprints a chat conversation so successive turns of the same agent
// map to the same admission session. The system prompt and the first user message are
// replayed verbatim on every turn while later messages grow, so the first two messages
// are the stable prefix. The model and the caller's identity are part of the key: the
// same opening from two callers, or against two models, is two sessions.
//
// Distinct conversations of one caller that share both opening messages collapse into
// one session; they share cached blocks up to the common prefix anyway.
//
// Returns empty (no session: charged while running, holds nothing) when there are no
// messages.
func sessionKey(model, owner string, body map[string]any) string {
	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) == 0 {
		return ""
	}
	h := sha256.New()
	io.WriteString(h, model+"\x00"+owner+"\x00")
	enc := json.NewEncoder(h) // deterministic: encoding/json sorts map keys
	for _, m := range msgs[:min(2, len(msgs))] {
		if err := enc.Encode(m); err != nil {
			return ""
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
