# Multi-stage build: compile in a Go image, run in distroless.
# Result is ~15 MB. No package manager, no shell, no surprise CVEs.

FROM golang:1.23-alpine AS build
WORKDIR /src
# Cache deps separately from source so `docker build` only re-downloads
# modules when go.mod/go.sum change.
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
COPY kvmirror ./kvmirror
# Static binary: no cgo, no glibc dependency. Trimpath strips source
# paths so the binary is reproducible regardless of build host.
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/cavp ./cmd/cavp

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/cavp /cavp
# Config path matches the compose volume mount. Override with -config
# if running outside compose.
EXPOSE 4001
USER nonroot:nonroot
ENTRYPOINT ["/cavp"]
CMD ["-config", "/etc/cavp/config.yaml"]
