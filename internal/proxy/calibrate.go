package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// calibrate measures prefill_overhead_blocks and decode_overhead_blocks against an idle
// engine: the blocks each running request takes beyond its prompt and output.
//
// It sends 1, 2 and 4 concurrent requests straight to the engine (not through the proxy),
// each a random token-id prompt spanning at least two blocks, and polls kv_cache_usage_perc.
// Overhead per request is the usage above the round's starting baseline, minus the attention
// blocks the requests hold at that moment (from their streamed token counts), divided by the
// request count. The decode figure is the peak once every request decodes, in any round. The
// prefill figure comes from the single-request round only, counting the request at its whole
// prompt from when it runs: with several requests, or a prompt longer than one prefill step,
// the engine holds less than that while prefilling and the figure would read low. It is
// reported as at least the decode figure. Only that round is required: the concurrent ones
// refine the decode figure and are skipped, with the reason, when they cannot be measured.
//
// It needs the engine to itself: nothing else running or waiting, before or during a round,
// and no inference proxy in front of it admitting traffic (stop the proxy: calibration
// traffic would also trip its overcommit guard). Pinned blocks of finished turns (grace pins in
// halt95's vLLM fork, v2.5) are part of each round's baseline; they must not be released mid-round, so run
// it with plenty of free KV space, ideally right after the engine starts.
//
// It refuses rather than report a figure it could not check: a data-parallel engine (more
// than one series of a gauge: its pool is per engine), metrics without the usage, running
// or waiting gauge its checks read, and a round in which the usage gauge never moved.
func calibrate(cfg *Config, out io.Writer) error {
	ctx := context.Background()
	up := strings.TrimRight(cfg.Upstream, "/")
	c := &calClient{up: up, http: &http.Client{Timeout: 10 * time.Minute}, key: os.Getenv("VLLM_API_KEY")}

	blocks, blockSize, err := fetchEnginePool(ctx, up)
	if err != nil {
		return fmt.Errorf("engine pool: %w", err)
	}
	model, err := c.model(ctx)
	if err != nil {
		return err
	}
	// At least two blocks, so overhead that grows along a prompt (checkpoints at block
	// boundaries) shows; capped so a round stays small.
	const decodeTokens = 512
	prompt := min(max(1536, 2*blockSize+256), 16384)
	perReq := (prompt + decodeTokens + blockSize - 1) / blockSize // most attention blocks a request holds
	usable := blocks - 1
	fmt.Fprintf(out, "engine %s: %d blocks of %d tokens; each request: %d prompt + %d output tokens (up to %d attention blocks)\n",
		model, usable, blockSize, prompt, decodeTokens, perReq)

	var maxPrefill, maxDecode float64
	for _, n := range []int{1, 2, 4} {
		p, d, err := c.round(ctx, model, n, prompt, decodeTokens, perReq, blockSize, usable)
		if err != nil && n == 1 {
			return fmt.Errorf("1 request: %w", err)
		}
		if err != nil {
			// The single-request round gave both figures; concurrent rounds only refine the
			// decode one, and some engines move pinned blocks when requests queue for prefill.
			fmt.Fprintf(out, "  %d concurrent: skipped: %v\n", n, err)
			continue
		}
		if n == 1 {
			maxPrefill = p
			fmt.Fprintf(out, "  1 request: prefill %.1f, decode %.1f blocks\n", p, d)
		} else {
			fmt.Fprintf(out, "  %d concurrent: decode %.1f blocks per request\n", n, d)
		}
		maxDecode = max(maxDecode, d)
	}
	maxDecode = max(maxDecode, 0)
	suggest := func(v float64) int { return int(v+0.999) + 2 } // round up, plus 2 blocks of margin
	fmt.Fprintf(out, "\nadmission:\n  cache:\n    prefill_overhead_blocks: %d\n    decode_overhead_blocks: %d\n",
		suggest(max(maxPrefill, maxDecode)), suggest(maxDecode))
	fmt.Fprintf(out, "\nMeasured with %d-token prompts. On some engines the overhead grows with prompt length\n"+
		"(checkpoints along a long prompt): for long-context traffic, prefer a higher figure and\n"+
		"watch the overcommit guard.\n", prompt)
	return nil
}

// calHeadroom bounds the overhead a request may take during calibration.
const calHeadroom = 32

type calClient struct {
	up, key string
	http    *http.Client
}

func (c *calClient) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var r io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.up+path, r)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.key != "" {
		req.Header.Set("Authorization", "Bearer "+c.key)
	}
	return c.http.Do(req)
}

func (c *calClient) model(ctx context.Context) (string, error) {
	resp, err := c.do(ctx, http.MethodGet, "/v1/models", nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var m struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil || len(m.Data) == 0 {
		return "", fmt.Errorf("engine /v1/models: no model (%v)", err)
	}
	return m.Data[0].ID, nil
}

// state reads usage (in blocks) and the running and waiting counts.
func (c *calClient) state(ctx context.Context, usable int) (used float64, busy int, err error) {
	used, running, waiting, err := c.stateDetail(ctx, usable)
	return used, running + waiting, err
}

func (c *calClient) stateDetail(ctx context.Context, usable int) (used float64, running, waiting int, err error) {
	resp, err := c.do(ctx, http.MethodGet, "/metrics", nil)
	if err != nil {
		return 0, 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, 0, fmt.Errorf("metrics: %s", resp.Status)
	}
	// Every check of a round reads these three gauges, labelled or not; one series each.
	names := []string{"vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"}
	var vals [3]float64
	var series [3]int
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		for i, name := range names {
			if _, v, ok := metricValue(sc.Text(), name); ok {
				vals[i] = v
				series[i]++
			}
		}
	}
	if err := sc.Err(); err != nil {
		return 0, 0, 0, err
	}
	for i, name := range names {
		switch {
		case series[i] == 0:
			return 0, 0, 0, fmt.Errorf("the engine's metrics have no %s: calibrate cannot run its checks without it", name)
		case series[i] > 1:
			return 0, 0, 0, fmt.Errorf("the engine's metrics have %d %s series (data-parallel?): calibrate measures a single engine", series[i], name)
		}
	}
	return vals[0] * float64(usable), int(vals[1]), int(vals[2]), nil
}

func (c *calClient) round(ctx context.Context, model string, n, prompt, decode, perReq, blockSize, usable int) (prefill, dec float64, err error) {
	base, busy, err := c.state(ctx, usable)
	if err != nil {
		return 0, 0, err
	}
	if busy > 0 {
		return 0, 0, fmt.Errorf("engine is not idle (%d running or waiting); calibrate needs it to itself", busy)
	}
	// Room for the round with up to calHeadroom overhead blocks each, so the engine has no
	// reason to release pinned blocks during it.
	if need := float64(n * (perReq + calHeadroom)); float64(usable)-base < need {
		return 0, 0, fmt.Errorf("%.0f blocks free, a round of %d needs %.0f: pinned blocks of finished turns hold the rest; run right after the engine starts", float64(usable)-base, n, need)
	}
	generated := make([]atomic.Int64, n) // -1 until the first token
	for i := range generated {
		generated[i].Store(-1)
	}
	var wg sync.WaitGroup
	errs := make(chan error, n)
	var done atomic.Int32
	for i := 0; i < n; i++ {
		ids := make([]int, prompt)
		for j := range ids {
			ids[j] = 1000 + rand.Intn(20000) // random ids: no prefix-cache hit
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			defer done.Add(1)
			errs <- c.stream(ctx, model, ids, decode, &generated[i])
		}(i)
	}
	prefill, dec = -1e9, -1e9
	peak := base
	// Sample only while all n run: a finished request's blocks go back (or into a pin that
	// may push out an older one), which says nothing about overhead.
	for done.Load() == 0 {
		used, running, waiting, err := c.stateDetail(ctx, usable)
		if err != nil {
			return 0, 0, fmt.Errorf("reading the engine's metrics mid-round: %w", err)
		}
		if running+waiting > n {
			return 0, 0, fmt.Errorf("other requests reached the engine during the round (%d running, %d waiting)", running, waiting)
		}
		peak = max(peak, used)
		if used < base-0.5 {
			return 0, 0, fmt.Errorf("usage fell below the round's baseline (%.1f < %.1f blocks): the engine released pinned blocks mid-round (other traffic resuming a conversation, or a shortage of free blocks); rerun when the engine is quiet, ideally right after it starts", used, base)
		}
		all, attn, answered := true, 0, 0
		for i := range generated {
			g := generated[i].Load()
			if g < 0 {
				all = false
				continue
			}
			answered++
			attn += (prompt + int(g) + blockSize - 1) / blockSize
		}
		// Running but not answered yet: its whole prompt (one prefill step).
		attn += max(running-answered, 0) * ((prompt + blockSize - 1) / blockSize)
		per := (used - base - float64(attn)) / float64(n)
		if all {
			dec = max(dec, per)
		} else {
			prefill = max(prefill, per)
		}
		time.Sleep(100 * time.Millisecond)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		if e != nil {
			return 0, 0, e
		}
	}
	// The engine's metrics may still count the round's own requests for a moment.
	for i := 0; ; i++ {
		_, busy, err := c.state(ctx, usable)
		if err == nil && busy == 0 {
			break
		}
		if i == 30 {
			return 0, 0, fmt.Errorf("the engine stayed busy after the round: other traffic")
		}
		time.Sleep(100 * time.Millisecond)
	}
	if peak-base < 1 {
		return 0, 0, fmt.Errorf("the engine's KV usage never rose above its baseline during the round (%.1f blocks): the gauge does not reflect the requests, so nothing was measured", base)
	}
	if dec < -1e8 {
		return 0, 0, fmt.Errorf("no sample with every request decoding (the round was too fast to observe)")
	}
	return max(prefill, dec), dec, nil
}

// stream sends one completion and keeps generated at its output token count (-1 until the
// first token).
func (c *calClient) stream(ctx context.Context, model string, ids []int, decode int, generated *atomic.Int64) error {
	resp, err := c.do(ctx, http.MethodPost, "/v1/completions", map[string]any{
		"model": model, "prompt": ids, "max_tokens": decode, "min_tokens": decode,
		"ignore_eos": true, "stream": true, "temperature": 0,
		"stream_options": map[string]any{"include_usage": true, "continuous_usage_stats": true},
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("completion: %s: %s", resp.Status, b)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	withUsage := 0
	for sc.Scan() {
		l, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || !strings.HasPrefix(l, "{") {
			continue
		}
		var ch struct {
			Usage *struct {
				CompletionTokens int64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(l), &ch) == nil && ch.Usage != nil {
			withUsage++
			generated.Store(ch.Usage.CompletionTokens)
		} else if generated.Load() < 0 {
			generated.Store(0)
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if withUsage < 2 && decode > 1 {
		return fmt.Errorf("the engine does not report token counts while streaming (stream_options.continuous_usage_stats); calibrate needs them")
	}
	return nil
}
