# WGPU picks the wgpu-native binding. wgpu24 binds the v29 C API, which
# can see non-conformant Vulkan drivers -- that is what reaches the real
# GPU through dozen inside WSL2, where a plain wgpu build falls back to a
# software rasterizer. Build the other one with "make WGPU=wgpu" when a
# driver disagrees with v29, and pair the binary with the matching
# library. See docs/guide/gpu.md.
WGPU ?= wgpu24
BIN := tensai
# Windows will not run a file without the extension, so the local build
# carries it; BIN itself stays bare because it also names the release
# archives, which are cross-built from Linux.
ifeq ($(OS),Windows_NT)
EXE := .exe
RM_F := del /q
RM_RF := rmdir /s /q
else
EXE :=
RM_F := rm -f
RM_RF := rm -rf
endif
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
	go build -tags $(WGPU) -ldflags=$(BUILD_LDFLAGS) -o $(BIN)$(EXE) ./cmd/tensai

.PHONY: install
install:
	go install -tags $(WGPU) -ldflags=$(BUILD_LDFLAGS) ./cmd/tensai

.PHONY: show-version
show-version: $(GOBIN)/gobump
	gobump show -r ./cmd/tensai

$(GOBIN)/gobump:
	go install github.com/x-motemen/gobump/cmd/gobump@latest

.PHONY: cross
# Both bindings ship: tensai_* is the wgpu24 build, which wants a
# v29-series libwgpu_native, and tensai-wgpu22_* the v22 one. Neither
# archive carries the library, so a driver that disagrees with one of them
# would otherwise leave a user with no binary and a Go toolchain to
# install. The binary inside is called tensai either way.
cross: $(GOBIN)/goxz
	goxz -n $(BIN) -pv=v$(VERSION) -os linux,darwin,windows -arch amd64,arm64 \
		-build-tags wgpu24 -build-ldflags=$(BUILD_LDFLAGS) ./cmd/tensai
	goxz -n $(BIN)-wgpu22 -o $(BIN) -pv=v$(VERSION) -os linux,darwin -arch amd64,arm64 \
		-build-tags wgpu -build-ldflags=$(BUILD_LDFLAGS) ./cmd/tensai
	goxz -n $(BIN)-wgpu22 -o $(BIN).exe -pv=v$(VERSION) -os windows -arch amd64,arm64 \
		-build-tags wgpu -build-ldflags=$(BUILD_LDFLAGS) ./cmd/tensai

$(GOBIN)/goxz:
	go install github.com/Songmu/goxz/cmd/goxz@latest

.PHONY: test
test:
	go test ./...

.PHONY: clean
clean:
	-$(RM_F) $(BIN)$(EXE)
	-$(RM_RF) goxz
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
	@gobump patch -w ./cmd/tensai
	git commit -am "Bump up version to $(VERSION)"
	git tag "v$(VERSION)"
	git push origin main
	git push origin "refs/tags/v$(VERSION)"

.PHONY: upload
upload: $(GOBIN)/ghr
	ghr -body="" "v$(VERSION)" goxz

$(GOBIN)/ghr:
	go install github.com/tcnksm/ghr@latest
