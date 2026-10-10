// Package proxy is cache-aware-vllm-proxy: an OpenAI-compatible reverse proxy that
// admits requests to one vLLM engine by what its prefix cache holds (docs/design.md).
//
// Main parses flags, loads the config, starts the background loops (model resolver,
// readiness probe, admission, KV event mirror, engine pool and usage), listens on
// cfg.Listen, and on SIGTERM/SIGINT drains for 10 s.
package proxy

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mgabor3141/cache-aware-vllm-proxy/kvmirror"
)

// Main runs the proxy: flags, config, then serve until a signal.
func Main() {
	configPath := flag.String("config", "/etc/cavp/config.yaml", "path to config.yaml")
	check := flag.Bool("check", false, "load and validate the config, then exit")
	calib := flag.Bool("calibrate", false, "measure the per-request KV overhead on the idle upstream engine, print config values, exit")
	flag.Parse()

	cfg, err := LoadConfig(*configPath)
	if err != nil {
		fatal("config: %v", err)
	}
	if *calib {
		if err := calibrate(cfg, os.Stdout); err != nil {
			fatal("calibrate: %v", err)
		}
		return
	}
	if *check {
		fmt.Println("config ok:", *configPath)
		return
	}

	// Single HTTP client shared by the resolver and the reverse proxy's
	// internal Transport. The resolver doesn't need streaming semantics
	// so a default client is fine.
	resolver := NewModelResolver(cfg.Upstream, &http.Client{Timeout: 15 * time.Second})

	logger := NewEventLogger(os.Stdout)

	rootCtx, rootCancel := context.WithCancel(context.Background())
	defer rootCancel()

	prx, err := NewProxy(cfg, resolver, logger)
	if err != nil {
		fatal("proxy init: %v", err)
	}
	go prx.admission.Run(rootCtx)
	ep := cfg.Admission.Cache.KVEventsEndpoint
	go prx.admission.mirror.Run(rootCtx, ep, kvmirror.ReplayEndpoint(ep))
	go runEnginePool(rootCtx, cfg.Upstream, prx.admission)
	go runEngineUsage(rootCtx, cfg.Upstream, prx.admission)

	go resolver.RunRefreshLoop(rootCtx, cfg.ModelsRefreshInterval, func(err error) {
		// Operator-visible but not noisy. The next successful refresh
		// will silently restore service; only the gap is worth reporting.
		log.Printf("level=warn msg=\"models refresh failed\" err=%q", err.Error())
	})
	go prx.health.Run(rootCtx)

	mux := http.NewServeMux()
	mux.Handle("/", prx)
	// Tiny liveness for compose healthchecks. Doesn't touch upstream;
	// returning 200 only proves the proxy process is alive and serving.
	// Readiness (upstream reachable + model resolved) lives at /readyz
	// for future use; not wired into compose yet.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !prx.health.Ready() {
			http.Error(w, "inference upstream is not ready", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})

	srv := &http.Server{
		Addr:    cfg.Listen,
		Handler: mux,
		// No request deadline: a request lasts until it finishes or its
		// client disconnects (agents retry on their own timeouts).
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: streaming responses can legitimately last
		// minutes; the request-scoped context handles cancellation.
	}

	log.Printf("level=info msg=\"cavp listening\" addr=%q upstream=%q aliases=%d auth=%q",
		cfg.Listen, cfg.Upstream, len(cfg.Aliases), cfg.Auth.Mode)

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	select {
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			fatal("listen: %v", err)
		}
	case sig := <-sigCh:
		log.Printf("level=info msg=\"shutting down\" signal=%q", sig)
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_ = srv.Shutdown(shutCtx)
		cancel()
		rootCancel()
	}
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "level=fatal msg=%q\n", fmt.Sprintf(format, a...))
	os.Exit(1)
}
