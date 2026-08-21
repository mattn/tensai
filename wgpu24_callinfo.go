//go:build wgpu24 && (linux || darwin)

package tensai

import "github.com/ebitengine/purego"

// The three entry points that take their callback-info struct by value.
// SysV and AAPCS hand a 40-byte aggregate to the callee in registers and
// on the stack, which purego reproduces from the Go struct type, so these
// signatures mirror webgpu.h directly. Windows needs the same calls spelled
// differently; see wgpu24_callinfo_windows.go.
var (
	fnInstanceRequestAdapter func(uintptr, *wgpuRequestAdapterOptions, wgpuCallbackInfo) uint64
	fnAdapterRequestDevice   func(uintptr, *wgpuDeviceDescriptor, wgpuCallbackInfo) uint64
	fnBufferMapAsync         func(uintptr, uint64, uintptr, uintptr, wgpuCallbackInfo) uint64
)

func registerCallInfoFns(lib uintptr) {
	purego.RegisterLibFunc(&fnInstanceRequestAdapter, lib, "wgpuInstanceRequestAdapter")
	purego.RegisterLibFunc(&fnAdapterRequestDevice, lib, "wgpuAdapterRequestDevice")
	purego.RegisterLibFunc(&fnBufferMapAsync, lib, "wgpuBufferMapAsync")
}

func requestAdapter(instance uintptr, opts *wgpuRequestAdapterOptions, info wgpuCallbackInfo) uint64 {
	return fnInstanceRequestAdapter(instance, opts, info)
}

func requestDevice(adapter uintptr, desc *wgpuDeviceDescriptor, info wgpuCallbackInfo) uint64 {
	return fnAdapterRequestDevice(adapter, desc, info)
}

func bufferMapAsync(buf uintptr, mode uint64, offset, size uintptr, info wgpuCallbackInfo) uint64 {
	return fnBufferMapAsync(buf, mode, offset, size, info)
}
