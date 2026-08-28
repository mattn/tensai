# wgpu24 binds the v29 wgpu-native C API, which can see non-conformant
# Vulkan drivers. That is what reaches the real GPU through dozen inside
# WSL2; a plain wgpu build falls back to a software rasterizer there.
# See docs/guide/gpu.md.
BIN := tensai
VERSION := $$(make -s show-version)
CURRENT_REVISION := $(shell git rev-parse --short HEAD)
BUILD_LDFLAGS := "-s -w -X main.revision=$(CURRENT_REVISION)"
GOBIN ?= $(shell go env GOPATH)/bin
export GOEXPERIMENT=simd
export CGO_ENABLED=0

.PHONY: all
all: clean build

.PHONY: build
build:
	go build -tags wgpu24 -ldflags=$(BUILD_LDFLAGS) -o $(BIN) ./cmd/tensai

.PHONY: install
install:
	go install -tags wgpu24 -ldflags=$(BUILD_LDFLAGS) ./cmd/tensai

.PHONY: show-version
show-version: $(GOBIN)/gobump
	gobump show -r ./cmd/tensai

$(GOBIN)/gobump:
	go install github.com/x-motemen/gobump/cmd/gobump@latest

.PHONY: cross
cross: $(GOBIN)/goxz
	goxz -n $(BIN) -pv=v$(VERSION) -os linux,darwin,windows -arch amd64,arm64 \
		-build-tags wgpu24 -build-ldflags=$(BUILD_LDFLAGS) ./cmd/tensai

$(GOBIN)/goxz:
	go install github.com/Songmu/goxz/cmd/goxz@latest

.PHONY: test
test:
	go test ./...

.PHONY: clean
clean:
	rm -rf $(BIN) goxz
	go clean

.PHONY: bump
bump: $(GOBIN)/gobump
ifneq ($(shell git status --porcelain),)
	$(error git workspace is dirty)
endif
ifneq ($(shell git rev-parse --abbrev-ref HEAD),main)
	$(error current branch is not main)
endif
	@gobump up -w ./cmd/tensai
	git commit -am "Bump up version to $(VERSION)"
	git tag "v$(VERSION)"
	git push origin main
	git push origin "refs/tags/v$(VERSION)"

.PHONY: bump-force
bump-force: $(GOBIN)/gobump
	@gobump patch -w .
	git commit -am "Bump up version to $(VERSION)"
	git tag "v$(VERSION)"
	git push origin main
	git push origin "refs/tags/v$(VERSION)"

.PHONY: upload
upload: $(GOBIN)/ghr
	ghr -body="" "v$(VERSION)" goxz

$(GOBIN)/ghr:
	go install github.com/tcnksm/ghr@latest
