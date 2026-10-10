package proxy

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// The engine's KV block pool, read from its Prometheus metrics instead of configured:
//
//	vllm:cache_config_info{block_size="3200",...,num_gpu_blocks="201",...} 1.0
//
// It changes only when the engine restarts with a different model or cache size, so a
// slow refresh is enough. Until it is known, nothing is admitted.

var (
	poolBlocksRe = regexp.MustCompile(`[{,]num_gpu_blocks="(\d+)"`)
	poolSizeRe   = regexp.MustCompile(`[{,]block_size="(\d+)"`)
)

func parseEnginePool(r io.Reader) (blocks, blockSize int, err error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		l := sc.Text()
		if len(l) < 23 || l[:23] != "vllm:cache_config_info{" {
			continue
		}
		b, s := poolBlocksRe.FindStringSubmatch(l), poolSizeRe.FindStringSubmatch(l)
		if b == nil || s == nil {
			return 0, 0, fmt.Errorf("cache_config_info without num_gpu_blocks or block_size")
		}
		blocks, _ = strconv.Atoi(b[1])
		blockSize, _ = strconv.Atoi(s[1])
		if blocks <= 0 || blockSize <= 0 {
			return 0, 0, fmt.Errorf("cache_config_info: %d blocks of %d", blocks, blockSize)
		}
		return blocks, blockSize, nil
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	return 0, 0, fmt.Errorf("no vllm:cache_config_info in metrics")
}

func fetchEnginePool(ctx context.Context, upstream string) (int, int, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, upstream+"/metrics", nil)
	if err != nil {
		return 0, 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("metrics: %s", resp.Status)
	}
	return parseEnginePool(resp.Body)
}

// metricValue parses one Prometheus text line of the named metric, labelled or not, into
// its label set and value.
func metricValue(line, name string) (labels string, v float64, ok bool) {
	rest, found := strings.CutPrefix(line, name)
	if !found {
		return "", 0, false
	}
	if strings.HasPrefix(rest, "{") {
		i := strings.IndexByte(rest, '}')
		if i < 0 {
			return "", 0, false
		}
		labels, rest = rest[:i+1], rest[i+1:]
	}
	f := strings.Fields(rest)
	if len(f) == 0 || !strings.HasPrefix(rest, " ") {
		return "", 0, false // another metric sharing the prefix
	}
	v, err := strconv.ParseFloat(f[0], 64)
	return labels, v, err == nil
}

// runEngineUsage feeds q the engine's occupied blocks and preemption counters every
// second. Missing reads leave the last value; the model's own charges stay the floor
// either way.
func runEngineUsage(ctx context.Context, upstream string, q *AdmissionQueue) {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		if u, pre, ok := fetchEngineUsage(ctx, upstream); ok {
			q.EngineSignals(pre)
			q.mu.Lock()
			pool := q.pool
			q.mu.Unlock()
			q.SetEngineUsed(int(u * float64(pool)))
		}
	}
}

// fetchEngineUsage reads the engine's KV usage (the highest, should there be several
// series) and its preemption counter per series. ok is false when the metrics could not
// be read or carry no usage.
func fetchEngineUsage(ctx context.Context, upstream string) (usage float64, preemptions map[string]float64, ok bool) {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, upstream+"/metrics", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, nil, false
	}
	preemptions = map[string]float64{}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		if _, v, found := metricValue(sc.Text(), "vllm:kv_cache_usage_perc"); found {
			usage, ok = max(usage, v), true
		} else if labels, v, found := metricValue(sc.Text(), "vllm:num_preemptions_total"); found {
			preemptions[labels] = v
		}
	}
	return usage, preemptions, ok
}

// runEnginePool keeps q's pool current: every 10 s until known, then every minute. A
// failed read keeps the last value (the engine restarting looks like this).
func runEnginePool(ctx context.Context, upstream string, q *AdmissionQueue) {
	var last [2]int
	for {
		wait := time.Minute
		blocks, size, err := fetchEnginePool(ctx, upstream)
		switch {
		case err != nil:
			if last[0] == 0 {
				wait = 10 * time.Second
				log.Printf("level=warn msg=\"engine KV pool unknown, requests wait\" err=%q", err.Error())
			}
		case [2]int{blocks, size} != last:
			last = [2]int{blocks, size}
			q.SetPool(blocks, size)
			log.Printf("level=info msg=\"engine KV pool\" blocks=%d block_size=%d tokens=%d", blocks, size, (blocks-1)*size)
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
}
