// Package proxy is cache-aware-vllm-proxy: an OpenAI-compatible reverse proxy that
// admits requests to one vLLM engine by what its prefix cache holds (docs/design.md).
//
// Main parses flags, loads the config, starts the background loops (readiness probe,
// admission, KV event mirror, engine pool and usage), and serves on cfg.Listen. On
// SIGTERM/SIGINT, waiting requests are answered 503 and running ones get 10 s to finish.
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

	rootCtx, rootCancel := context.WithCancel(context.Background())
	defer rootCancel()

	prx, err := NewProxy(cfg, NewEventLogger(os.Stdout))
	if err != nil {
		fatal("proxy init: %v", err)
	}
	go prx.admission.Run(rootCtx)
	ep := cfg.Admission.Cache.KVEventsEndpoint
	go prx.admission.mirror.Run(rootCtx, ep, kvmirror.ReplayEndpoint(ep))
	go runEnginePool(rootCtx, cfg.Upstream, prx.admission)
	go runEngineUsage(rootCtx, cfg.Upstream, prx.admission)
	go prx.ready.Run(rootCtx)

	srv := &http.Server{
		Addr:    cfg.Listen,
		Handler: prx,
		// No request deadline: a request lasts until it finishes or its client leaves.
		// Writes are bounded per write (writeStallTimeout), headers here.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       5 * time.Minute,
	}

	log.Printf("level=info msg=\"cavp listening\" addr=%q upstream=%q auth=%q max_active=%d kv_events=%q",
		cfg.Listen, cfg.Upstream, cfg.Auth.Mode, cfg.Admission.MaxActive, cfg.Admission.Cache.KVEventsEndpoint)

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
		prx.Close() // waiting requests answer 503 and leave
		shutCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := srv.Shutdown(shutCtx); err != nil {
			log.Printf("level=warn msg=\"requests still running at shutdown were cut\" err=%q", err.Error())
			srv.Close()
		}
		cancel()
		rootCancel()
	}
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "level=fatal msg=%q\n", fmt.Sprintf(format, a...))
	os.Exit(1)
}
