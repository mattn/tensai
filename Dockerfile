# A scratch image around the static tensai binary:
#
#	docker run -it --rm ghcr.io/mattn/tensai chat -q8
#
# Models download into /cache (XDG_CACHE_HOME), so mount a volume — or
# the host's own cache — to keep them across runs:
#
#	docker run -it --rm -v tensai-cache:/cache ghcr.io/mattn/tensai chat -q8
FROM golang:1.27 AS build
ARG TARGETARCH
ARG REVISION=docker
WORKDIR /src
COPY . .
# No wgpu tag: purego's dlopen layer needs a dynamic libc, and scratch
# has neither that nor Vulkan — the container is the CPU path, and the
# untagged build is truly static.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$TARGETARCH GOEXPERIMENT=simd \
    go build -ldflags "-s -w -X main.revision=$REVISION" \
    -o /tensai ./cmd/tensai

FROM scratch
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=build /tensai /tensai
ENV XDG_CACHE_HOME=/cache
VOLUME /cache
ENTRYPOINT ["/tensai"]
CMD ["chat"]
