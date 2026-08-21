#!/bin/bash
# test-wgpu24.sh - exercise the -tags wgpu24 backend (new wgpu-native C API),
# including the real host GPU from inside WSL2 via Mesa's dozen driver.
#
# What it does, idempotently and without root:
#   1. fetches wgpu-native v29 into  ~/.local/lib/wgpu29/libwgpu_native.so
#   2. fetches Mesa's dzn driver     ~/.local/lib/dzn/libvulkan_dzn.so
#      (extracted from the kisak-mesa PPA .deb; the distro mesa lacks dzn)
#      and writes an ICD manifest    ~/.local/lib/dzn/dzn_icd.json
#   3. runs the GPU tests and the example sweep on the host GPU through
#      dozen.
#
# dzn is not a conformant Vulkan implementation; wgpu only accepts it under
# the wgpu24 bindings, which set AllowUnderlyingNoncompliantAdapter. Expect
# the "testing use only" warning. On machines with a system-wide Vulkan GPU
# driver (native Linux), skip the dzn parts and just run the tests.
set -euo pipefail
cd "$(dirname "$0")"

WGPU_VER=v29.0.1.1
WGPU_URL=https://github.com/gfx-rs/wgpu-native/releases/download/$WGPU_VER/wgpu-linux-x86_64-release.zip
# Ubuntu noble build of Mesa that ships the dzn driver. Override when the
# PPA rotates versions: https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa
MESA_DEB_URL=${MESA_DEB_URL:-https://launchpad.net/~kisak/+archive/ubuntu/kisak-mesa/+files/mesa-vulkan-drivers_26.1.7~kisak1~n_amd64.deb}

LIBDIR=$HOME/.local/lib
WGPU29=$LIBDIR/wgpu29/libwgpu_native.so
DZN_DIR=$LIBDIR/dzn
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "wgpu-native $WGPU_VER -> $WGPU29"
if [ ! -f "$WGPU29" ]; then
	curl -sL -o "$TMP/wgpu.zip" "$WGPU_URL"
	mkdir -p "$(dirname "$WGPU29")"
	unzip -j -o -q "$TMP/wgpu.zip" lib/libwgpu_native.so -d "$(dirname "$WGPU29")"
fi
echo "ok"

step "Mesa dzn driver -> $DZN_DIR"
if [ ! -f "$DZN_DIR/dzn_icd.json" ]; then
	curl -sL -o "$TMP/mesa.deb" "$MESA_DEB_URL"
	dpkg -x "$TMP/mesa.deb" "$TMP/mesa"
	mkdir -p "$DZN_DIR"
	cp "$TMP/mesa/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so" "$DZN_DIR/"
	sed "s|\"library_path\": \"[^\"]*\"|\"library_path\": \"$DZN_DIR/libvulkan_dzn.so\"|" \
		"$TMP/mesa/usr/share/vulkan/icd.d/dzn_icd.json" > "$DZN_DIR/dzn_icd.json"
fi
# dzn links libdisplay-info.so.1; take it from the archive when the system
# lacks it (or: sudo apt install libdisplay-info1).
if ! ldconfig -p | grep -q libdisplay-info.so.1 && [ ! -f "$DZN_DIR/libdisplay-info.so.1" ]; then
	(cd "$TMP" && apt-get download libdisplay-info1 >/dev/null && dpkg -x libdisplay-info1_*.deb di)
	cp "$TMP"/di/usr/lib/x86_64-linux-gnu/libdisplay-info.so.1* "$DZN_DIR/"
fi
export LD_LIBRARY_PATH=$DZN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "ok"

export TENSAI_WGPU_LIB=$WGPU29

# -count=1: VK_DRIVER_FILES is read by the C library, not by Go, so the
# test cache cannot tell runs on different Vulkan drivers apart.
step "GPU tests on the host GPU via dozen"
VK_DRIVER_FILES=$DZN_DIR/dzn_icd.json go test -tags wgpu24 -count=1 -run TestGPU -v .

step "sweep on the host GPU: AVX2 kernel vs GPU"
VK_DRIVER_FILES=$DZN_DIR/dzn_icd.json GOEXPERIMENT=simd \
	go run -tags wgpu24 ./_example/wgpu -sweep
