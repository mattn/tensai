//go:build wgpu24 && windows

package tensai

import (
	"runtime"

	"github.com/ebitengine/purego"
)

// The same three entry points as wgpu24_callinfo.go, spelled for the
// Windows x64 convention. There an aggregate that is not 1, 2, 4, or 8
// bytes wide is passed by reference: the caller keeps a copy alive for the
// duration of the call and hands over its address. wgpuCallbackInfo is 40
// bytes, so a pointer parameter *is* the by-value ABI here, and purego's
// register-level struct support is not needed at all.
//
// The WGPUFuture results stay uint64: an 8-byte aggregate comes back in
// RAX on Windows exactly as it does under SysV.
var (
	fnInstanceRequestAdapter func(uintptr, *wgpuRequestAdapterOptions, *wgpuCallbackInfo) uint64
	fnAdapterRequestDevice   func(uintptr, *wgpuDeviceDescriptor, *wgpuCallbackInfo) uint64
	fnBufferMapAsync         func(uintptr, uint64, uintptr, uintptr, *wgpuCallbackInfo) uint64
)

func registerCallInfoFns(lib uintptr) {
	purego.RegisterLibFunc(&fnInstanceRequestAdapter, lib, "wgpuInstanceRequestAdapter")
	purego.RegisterLibFunc(&fnAdapterRequestDevice, lib, "wgpuAdapterRequestDevice")
	purego.RegisterLibFunc(&fnBufferMapAsync, lib, "wgpuBufferMapAsync")
}

func requestAdapter(instance uintptr, opts *wgpuRequestAdapterOptions, info wgpuCallbackInfo) uint64 {
	f := fnInstanceRequestAdapter(instance, opts, &info)
	runtime.KeepAlive(&info)
	return f
}

func requestDevice(adapter uintptr, desc *wgpuDeviceDescriptor, info wgpuCallbackInfo) uint64 {
	f := fnAdapterRequestDevice(adapter, desc, &info)
	runtime.KeepAlive(&info)
	return f
}

func bufferMapAsync(buf uintptr, mode uint64, offset, size uintptr, info wgpuCallbackInfo) uint64 {
	f := fnBufferMapAsync(buf, mode, offset, size, &info)
	runtime.KeepAlive(&info)
	return f
}
