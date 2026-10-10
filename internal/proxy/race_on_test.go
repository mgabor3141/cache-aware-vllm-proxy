//go:build race

package proxy

// raceEnabled: the race detector inflates allocations, so memory measurements skip.
const raceEnabled = true
