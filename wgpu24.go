//go:build wgpu24 && (linux || darwin || windows)

package tensai

// The WebGPU compute backend against the reworked wgpu-native C API
// (v24 and later; these bindings target the v29.0.1.1 release). Build with
// -tags wgpu24 instead of -tags wgpu; the exported surface (OpenGPU, GPU,
// GPUPower) is identical, so everything else — including _example/wgpu —
// works unchanged.
//
// What justifies the second binding generation is
// WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter, which un-hides
// non-conformant Vulkan drivers — concretely Mesa's dozen
// (Vulkan-on-D3D12), the only route to the real GPU inside WSL2.
//
// The new API passes WGPUStringView and callback-info structs by value.
// Every such struct here is reached through a pointer field except the
// three callback-info arguments, whose calling convention differs between
// SysV/AAPCS and Windows x64; wgpu24_callinfo.go and its _windows sibling
// hold that difference, and nothing else in this file has to care.
//
// Use a wgpu-native v29-series shared library with this build; the v22
// library will fail symbol registration cleanly.

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// GPU is a WebGPU device with a compiled matmul pipeline. Open one with
// OpenGPU, use it from any goroutine (calls are serialized internally), and
// Close it when done.
type GPU struct {
	mu         sync.Mutex
	instance   uintptr
	adapter    uintptr
	device     uintptr
	queue      uintptr
	module     uintptr
	module2    uintptr // optional integer-dot module; 0 if unsupported
	hasIntDot  bool
	pipes      gpuPipelines
	readback   gpuReadbackBuffer
	pool       gpuBufferPool
	bgCache    map[bgKey]uintptr
	batchEnc   uintptr // open command encoder while batching, see BeginBatch
	batchPass  uintptr // open compute pass inside the batch encoder
	name       string
	maxStorage uint64 // usable bytes per storage buffer, 0 = unknown
	closed     bool
}

// GPUPower tells OpenGPU which adapter to prefer on machines with more than
// one GPU. It is a hint: when only one adapter exists it is returned
// regardless.
type GPUPower uint32

const (
	GPUDefault         GPUPower = 0 // let the driver pick
	GPULowPower        GPUPower = 1 // prefer the integrated GPU
	GPUHighPerformance GPUPower = 2 // prefer the discrete GPU
)

// v29 API constants. Flags widened to 64 bits, statuses now start at 1.
const (
	wgpuSTypeShaderSourceWGSL = 0x00000002
	wgpuSTypeInstanceExtras   = 0x00030004

	wgpuBufferUsageMapRead uint64 = 0x01
	wgpuBufferUsageCopySrc uint64 = 0x04
	wgpuBufferUsageCopyDst uint64 = 0x08
	wgpuBufferUsageUniform uint64 = 0x40
	wgpuBufferUsageStorage uint64 = 0x80

	wgpuMapModeRead uint64 = 0x01

	wgpuStatusSuccess = 1 // request-adapter/device and map-async alike

	wgpuCallbackModeAllowProcessEvents = 2
	wgpuCallbackModeAllowSpontaneous   = 3

	wgpuInstanceFlagAllowNoncompliant uint64 = 1 << 3

	wgpuStrlen = ^uintptr(0) // WGPU_STRLEN: treat data as null-terminated
)

// Hand-mirrored v29 structs; field order, widths, and padding must match
// the 64-bit C layouts exactly.

type wgpuStringView struct {
	data   *byte
	length uintptr
}

// sv builds a WGPUStringView over a Go string. The returned backing slice
// must be kept alive across the C call.
func sv(s string) (wgpuStringView, []byte) {
	if s == "" {
		return wgpuStringView{}, nil
	}
	b := []byte(s)
	return wgpuStringView{data: &b[0], length: uintptr(len(b))}, b
}

type wgpuChainedStruct struct {
	next  uintptr
	sType uint32
	_     uint32
}

type wgpuInstanceExtras struct {
	chain        wgpuChainedStruct
	backends     uint64 // 0 = all
	flags        uint64
	dx12Compiler uint32
	gles3Minor   uint32
	glFence      uint32
	_            uint32
	dxcPath      wgpuStringView
	dxcMaxSM     uint32
	dx12Swap     uint32
	budgetCreate uintptr
	budgetLoss   uintptr
	dhType       uint32 // WGPUNativeDisplayHandle, union flattened
	_            uint32
	dh1, dh2     uintptr
}

type wgpuInstanceDescriptor struct {
	nextInChain          *wgpuInstanceExtras
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
}

type wgpuRequestAdapterOptions struct {
	nextInChain       uintptr
	featureLevel      uint32 // 0 = undefined, defaults to Core
	powerPreference   uint32
	forceFallback     uint32
	backendType       uint32
	compatibleSurface uintptr
}

// wgpuCallbackInfo is the shared {mode, callback, userdata1, userdata2}
// shape of WGPURequestAdapterCallbackInfo, WGPURequestDeviceCallbackInfo,
// and WGPUBufferMapCallbackInfo. It is passed to the C API by value.
type wgpuCallbackInfo struct {
	nextInChain uintptr
	mode        uint32
	_           uint32
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

type wgpuUncapturedErrorCallbackInfo struct {
	nextInChain uintptr
	callback    uintptr
	userdata1   uintptr
	userdata2   uintptr
}

// wgpuLimits mirrors the v29 WGPULimits: a chain pointer, 14 leading u32s,
// and three u64 sizes split by u32 runs (the tail gained maxImmediateSize
// and lost maxInterStageShaderComponents relative to v22, keeping 12
// u32s). Only the named sizes are read; the rest ride along so the
// adapter's limits can be requested back verbatim.
type wgpuLimits struct {
	nextInChain                 uintptr
	u32a                        [14]uint32
	maxUniformBufferBindingSize uint64
	maxStorageBufferBindingSize uint64
	u32b                        [3]uint32
	_                           uint32
	maxBufferSize               uint64
	u32c                        [12]uint32
}

type wgpuDeviceDescriptor struct {
	nextInChain          uintptr
	label                wgpuStringView
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
	queueNext            uintptr // WGPUQueueDescriptor, inlined
	queueLabel           wgpuStringView
	lost                 wgpuCallbackInfo // WGPUDeviceLostCallbackInfo, same shape
	uncaptured           wgpuUncapturedErrorCallbackInfo
}

type wgpuShaderSourceWGSL struct {
	chain wgpuChainedStruct
	code  wgpuStringView
}

type wgpuShaderModuleDescriptor struct {
	nextInChain *wgpuShaderSourceWGSL
	label       wgpuStringView
}

type wgpuComputeState struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    wgpuStringView
	constantCount uintptr
	constants     uintptr
}

type wgpuComputePipelineDescriptor struct {
	nextInChain uintptr
	label       wgpuStringView
	layout      uintptr // 0 = auto layout
	compute     wgpuComputeState
}

type wgpuBufferDescriptor struct {
	nextInChain      uintptr
	label            wgpuStringView
	usage            uint64
	size             uint64
	mappedAtCreation uint32
	_                uint32
}

type wgpuBindGroupEntry struct {
	nextInChain uintptr
	binding     uint32
	_           uint32
	buffer      uintptr
	offset      uint64
	size        uint64
	sampler     uintptr
	textureView uintptr
}

type wgpuBindGroupDescriptor struct {
	nextInChain uintptr
	label       wgpuStringView
	layout      uintptr
	entryCount  uintptr
	entries     *wgpuBindGroupEntry
}

type wgpuAdapterInfo struct {
	nextInChain  uintptr
	vendor       wgpuStringView
	architecture wgpuStringView
	device       wgpuStringView
	description  wgpuStringView
	backendType  uint32
	adapterType  uint32
	vendorID     uint32
	deviceID     uint32
	sgMin, sgMax uint32
}

// Function pointers resolved from the shared library. RequestAdapter,
// RequestDevice, and MapAsync are not here: they take a callback-info
// struct by value, so they live in the per-OS wgpu24_callinfo files behind
// the requestAdapter/requestDevice/bufferMapAsync wrappers.
var (
	fnCreateInstance          func(*wgpuInstanceDescriptor) uintptr
	fnAdapterGetInfo          func(uintptr, *wgpuAdapterInfo) uint32
	fnAdapterGetLimits        func(uintptr, *wgpuLimits) uint32
	fnDeviceGetQueue          func(uintptr) uintptr
	fnDeviceCreateShaderMod   func(uintptr, *wgpuShaderModuleDescriptor) uintptr
	fnDeviceCreatePipeline    func(uintptr, *wgpuComputePipelineDescriptor) uintptr
	fnPipelineGetLayout       func(uintptr, uint32) uintptr
	fnDeviceCreateBuffer      func(uintptr, *wgpuBufferDescriptor) uintptr
	fnDeviceCreateBindGroup   func(uintptr, *wgpuBindGroupDescriptor) uintptr
	fnDeviceCreateCmdEncoder  func(uintptr, unsafe.Pointer) uintptr
	fnEncoderBeginComputePass func(uintptr, unsafe.Pointer) uintptr
	fnPassSetPipeline         func(uintptr, uintptr)
	fnPassSetBindGroup        func(uintptr, uint32, uintptr, uintptr, unsafe.Pointer)
	fnPassDispatch            func(uintptr, uint32, uint32, uint32)
	fnPassEnd                 func(uintptr)
	fnEncoderCopyBuffer       func(uintptr, uintptr, uint64, uintptr, uint64, uint64)
	fnEncoderFinish           func(uintptr, unsafe.Pointer) uintptr
	fnQueueSubmit             func(uintptr, uintptr, unsafe.Pointer)
	fnQueueWriteBuffer        func(uintptr, uintptr, uint64, unsafe.Pointer, uintptr)
	fnBufferGetConstMapped    func(uintptr, uintptr, uintptr) unsafe.Pointer
	fnBufferUnmap             func(uintptr)
	fnDevicePoll              func(uintptr, uint32, unsafe.Pointer) uint32
	fnBufferRelease           func(uintptr)
	fnBindGroupRelease        func(uintptr)
	fnCmdBufferRelease        func(uintptr)
	fnCmdEncoderRelease       func(uintptr)
	fnPassRelease             func(uintptr)
	fnShaderModRelease        func(uintptr)
	fnLayoutRelease           func(uintptr)
	fnPipelineRelease         func(uintptr)
	fnQueueRelease            func(uintptr)
	fnDeviceRelease           func(uintptr)
	fnAdapterRelease          func(uintptr)
	fnInstanceRelease         func(uintptr)
)

// Callback plumbing, identical in spirit to the v22 file: NewCallback slots
// are permanent, created once, routed through globals under wgpuMu.
var (
	wgpuMu       sync.Mutex
	loadOnce     sync.Once
	loadErr      error
	cbAdapter    uintptr
	cbDevice     uintptr
	cbFailMsg    string
	cbMapDone    bool
	cbMapOK      bool
	uncapturedCB string

	adapterCB, deviceCB, mapCB, errorCB uintptr
)

// svString copies a WGPUStringView arriving from C as its two register
// halves (data pointer, length).
func svString(data, length uintptr) string {
	if data == 0 || length == 0 {
		return ""
	}
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&data))
	if length == wgpuStrlen {
		var out []byte
		for i := 0; ; i++ {
			c := *(*byte)(unsafe.Add(ptr, i))
			if c == 0 {
				return string(out)
			}
			out = append(out, c)
		}
	}
	return string(unsafe.Slice((*byte)(ptr), length))
}

func viewString(v wgpuStringView) string {
	return svString(uintptr(unsafe.Pointer(v.data)), v.length)
}

func libCandidates() []string {
	if p := os.Getenv("TENSAI_WGPU_LIB"); p != "" {
		return []string{p}
	}
	if runtime.GOOS == "darwin" {
		return []string{"libwgpu_native.dylib"}
	}
	return []string{"libwgpu_native.so"}
}

func loadWGPU() error {
	loadOnce.Do(func() {
		var lib uintptr
		var err error
		for _, name := range libCandidates() {
			lib, err = dlopenWGPU(name)
			if err == nil {
				break
			}
		}
		if lib == 0 {
			loadErr = fmt.Errorf("tensai: cannot load wgpu-native (set TENSAI_WGPU_LIB): %w", err)
			return
		}
		// Same symbol names, different ABI: calling a v22 library through
		// these bindings crashes, so gate on the packed version number.
		var fnGetVersion func() uint32
		purego.RegisterLibFunc(&fnGetVersion, lib, "wgpuGetVersion")
		if major := fnGetVersion() >> 24; major < 24 {
			loadErr = fmt.Errorf("tensai: wgpu-native v%d is too old for -tags wgpu24 (needs v24+); use -tags wgpu with it", major)
			return
		}
		for _, f := range []struct {
			ptr  any
			name string
		}{
			{&fnCreateInstance, "wgpuCreateInstance"},
			{&fnAdapterGetInfo, "wgpuAdapterGetInfo"},
			{&fnAdapterGetLimits, "wgpuAdapterGetLimits"},
			{&fnDeviceGetQueue, "wgpuDeviceGetQueue"},
			{&fnDeviceCreateShaderMod, "wgpuDeviceCreateShaderModule"},
			{&fnDeviceCreatePipeline, "wgpuDeviceCreateComputePipeline"},
			{&fnPipelineGetLayout, "wgpuComputePipelineGetBindGroupLayout"},
			{&fnDeviceCreateBuffer, "wgpuDeviceCreateBuffer"},
			{&fnDeviceCreateBindGroup, "wgpuDeviceCreateBindGroup"},
			{&fnDeviceCreateCmdEncoder, "wgpuDeviceCreateCommandEncoder"},
			{&fnEncoderBeginComputePass, "wgpuCommandEncoderBeginComputePass"},
			{&fnPassSetPipeline, "wgpuComputePassEncoderSetPipeline"},
			{&fnPassSetBindGroup, "wgpuComputePassEncoderSetBindGroup"},
			{&fnPassDispatch, "wgpuComputePassEncoderDispatchWorkgroups"},
			{&fnPassEnd, "wgpuComputePassEncoderEnd"},
			{&fnEncoderCopyBuffer, "wgpuCommandEncoderCopyBufferToBuffer"},
			{&fnEncoderFinish, "wgpuCommandEncoderFinish"},
			{&fnQueueSubmit, "wgpuQueueSubmit"},
			{&fnQueueWriteBuffer, "wgpuQueueWriteBuffer"},
			{&fnBufferGetConstMapped, "wgpuBufferGetConstMappedRange"},
			{&fnBufferUnmap, "wgpuBufferUnmap"},
			{&fnDevicePoll, "wgpuDevicePoll"},
			{&fnBufferRelease, "wgpuBufferRelease"},
			{&fnBindGroupRelease, "wgpuBindGroupRelease"},
			{&fnCmdBufferRelease, "wgpuCommandBufferRelease"},
			{&fnCmdEncoderRelease, "wgpuCommandEncoderRelease"},
			{&fnPassRelease, "wgpuComputePassEncoderRelease"},
			{&fnShaderModRelease, "wgpuShaderModuleRelease"},
			{&fnLayoutRelease, "wgpuBindGroupLayoutRelease"},
			{&fnPipelineRelease, "wgpuComputePipelineRelease"},
			{&fnQueueRelease, "wgpuQueueRelease"},
			{&fnDeviceRelease, "wgpuDeviceRelease"},
			{&fnAdapterRelease, "wgpuAdapterRelease"},
			{&fnInstanceRelease, "wgpuInstanceRelease"},
		} {
			purego.RegisterLibFunc(f.ptr, lib, f.name)
		}
		registerCallInfoFns(lib)

		// New-API callbacks receive WGPUStringView messages by value; on
		// SysV/AAPCS a 16-byte aggregate arrives as two register words, so
		// the Go side declares (data, length) pairs.
		adapterCB = purego.NewCallback(func(status uint32, adapter, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbAdapter = adapter
			} else {
				cbFailMsg = svString(msgData, msgLen)
			}
			return 0
		})
		deviceCB = purego.NewCallback(func(status uint32, device, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbDevice = device
			} else {
				cbFailMsg = svString(msgData, msgLen)
			}
			return 0
		})
		mapCB = purego.NewCallback(func(status uint32, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			cbMapDone = true
			cbMapOK = status == wgpuStatusSuccess
			if !cbMapOK && cbFailMsg == "" {
				cbFailMsg = svString(msgData, msgLen)
			}
			return 0
		})
		errorCB = purego.NewCallback(func(device uintptr, errType uint32, msgData, msgLen, ud1, ud2 uintptr) uintptr {
			if uncapturedCB == "" {
				uncapturedCB = svString(msgData, msgLen)
			}
			return 0
		})
	})
	return loadErr
}

// OpenGPU loads wgpu-native, picks an adapter, and compiles the matmul
// pipeline. On machines with several GPUs an optional GPUPower steers the
// choice: GPULowPower prefers the integrated GPU, GPUHighPerformance the
// discrete one. Non-conformant Vulkan drivers (Mesa's dozen inside WSL2)
// are allowed. The returned GPU is safe for concurrent use.
func OpenGPU(power ...GPUPower) (*GPU, error) {
	if err := loadWGPU(); err != nil {
		return nil, err
	}
	wgpuMu.Lock()
	defer wgpuMu.Unlock()

	g := &GPU{}
	extras := wgpuInstanceExtras{
		chain: wgpuChainedStruct{sType: wgpuSTypeInstanceExtras},
		flags: wgpuInstanceFlagAllowNoncompliant,
	}
	iDesc := wgpuInstanceDescriptor{nextInChain: &extras}
	g.instance = fnCreateInstance(&iDesc)
	runtime.KeepAlive(&extras)
	runtime.KeepAlive(&iDesc)
	if g.instance == 0 {
		return nil, errors.New("tensai: wgpuCreateInstance failed")
	}
	opts := wgpuRequestAdapterOptions{}
	if len(power) > 0 {
		opts.powerPreference = uint32(power[0])
	}
	// wgpu-native still fires these callbacks synchronously.
	cbAdapter, cbDevice, cbFailMsg = 0, 0, ""
	requestAdapter(g.instance, &opts, wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowSpontaneous, callback: adapterCB,
	})
	runtime.KeepAlive(&opts)
	if cbAdapter == 0 {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu adapter request failed: %s", cbFailMsg)
	}
	g.adapter = cbAdapter

	// The info strings are tiny and OpenGPU runs once per process, so they
	// are copied and never handed back to wgpuAdapterInfoFreeMembers.
	var info wgpuAdapterInfo
	fnAdapterGetInfo(g.adapter, &info)
	runtime.KeepAlive(&info)
	g.name = viewString(info.device)
	if g.name == "" {
		g.name = viewString(info.description)
	}
	switch info.adapterType {
	case 1:
		g.name += " (discrete)"
	case 2:
		g.name += " (integrated)"
	case 3:
		g.name += " (cpu)"
	}

	// Request the adapter's own limits back so big storage buffers are not
	// capped at the conservative defaults (128MiB bindings).
	var lim wgpuLimits
	dDesc := wgpuDeviceDescriptor{
		uncaptured: wgpuUncapturedErrorCallbackInfo{callback: errorCB},
	}
	if fnAdapterGetLimits(g.adapter, &lim) == 1 {
		lim.nextInChain = 0
		dDesc.requiredLimits = uintptr(unsafe.Pointer(&lim))
		g.maxStorage = min(lim.maxStorageBufferBindingSize, lim.maxBufferSize)
	}
	requestDevice(g.adapter, &dDesc, wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowSpontaneous, callback: deviceCB,
	})
	runtime.KeepAlive(&dDesc)
	runtime.KeepAlive(&lim)
	if cbDevice == 0 {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu device request failed: %s", cbFailMsg)
	}
	g.device = cbDevice
	g.queue = fnDeviceGetQueue(g.device)

	uncapturedCB = ""
	code, codeBuf := sv(matmulWGSL)
	wgsl := wgpuShaderSourceWGSL{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderSourceWGSL},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	g.module = fnDeviceCreateShaderMod(g.device, &smDesc)
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(codeBuf)
	if g.module == 0 || uncapturedCB != "" {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu shader compilation failed: %s", uncapturedCB)
	}
	if err := g.initPipelines(); err != nil {
		g.Close()
		return nil, err
	}
	return g, nil
}

// makeModuleFrom compiles an extra WGSL module, returning 0 on failure
// (the caller checks uncapturedCB for the reason).
func (g *GPU) makeModuleFrom(src string) uintptr {
	code, codeBuf := sv(src)
	wgsl := wgpuShaderSourceWGSL{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderSourceWGSL},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	m := fnDeviceCreateShaderMod(g.device, &smDesc)
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(codeBuf)
	return m
}

// makePipelineIn builds a compute pipeline for one entry point of an
// explicit module.
func (g *GPU) makePipelineIn(module uintptr, entry string) uintptr {
	e, eBuf := sv(entry)
	pDesc := wgpuComputePipelineDescriptor{
		compute: wgpuComputeState{module: module, entryPoint: e},
	}
	p := fnDeviceCreatePipeline(g.device, &pDesc)
	runtime.KeepAlive(&pDesc)
	runtime.KeepAlive(eBuf)
	return p
}

// makePipeline builds a compute pipeline for one entry point of g.module.
func (g *GPU) makePipeline(entry string) uintptr {
	e, eBuf := sv(entry)
	pDesc := wgpuComputePipelineDescriptor{
		compute: wgpuComputeState{module: g.module, entryPoint: e},
	}
	p := fnDeviceCreatePipeline(g.device, &pDesc)
	runtime.KeepAlive(&pDesc)
	runtime.KeepAlive(eBuf)
	return p
}

// Name reports the adapter wgpu selected, e.g. "Microsoft Direct3D12 (AMD
// Radeon(TM) Graphics) (integrated)" — handy for checking which GPU a power
// preference landed on.
func (g *GPU) Name() string { return g.name }

// Close releases the device and every object OpenGPU created. The GPU must
// not be used afterwards.
func (g *GPU) Close() {
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return
	}
	g.closed = true
	g.releaseReadback()
	g.releasePool()
	g.releasePipelines()
	for _, r := range []struct {
		fn func(uintptr)
		h  uintptr
	}{
		{fnShaderModRelease, g.module2},
		{fnShaderModRelease, g.module},
		{fnQueueRelease, g.queue},
		{fnDeviceRelease, g.device},
		{fnAdapterRelease, g.adapter},
		{fnInstanceRelease, g.instance},
	} {
		if r.h != 0 {
			r.fn(r.h)
		}
	}
}

// The primitives below back the shared GPUTensor layer in wgputensor.go;
// their signatures match the v22 bindings so that layer compiles against
// either API generation.

func (g *GPU) newBuffer(usage, size uint64) uintptr {
	desc := wgpuBufferDescriptor{usage: usage, size: size}
	buf := fnDeviceCreateBuffer(g.device, &desc)
	runtime.KeepAlive(&desc)
	return buf
}

func (g *GPU) makeBindGroup(layout uintptr, entries []wgpuBindGroupEntry) uintptr {
	desc := wgpuBindGroupDescriptor{layout: layout, entryCount: uintptr(len(entries)), entries: &entries[0]}
	bg := fnDeviceCreateBindGroup(g.device, &desc)
	runtime.KeepAlive(&desc)
	return bg
}

// mapRead blocks until buf (usage MapRead) is mapped and returns the mapped
// range; the caller must fnBufferUnmap when done.
func (g *GPU) mapRead(buf uintptr, bytes uint64) (unsafe.Pointer, error) {
	cbMapDone, cbMapOK, cbFailMsg = false, false, ""
	bufferMapAsync(buf, wgpuMapModeRead, 0, uintptr(bytes), wgpuCallbackInfo{
		mode: wgpuCallbackModeAllowProcessEvents, callback: mapCB,
	})
	for !cbMapDone {
		fnDevicePoll(g.device, 1, nil)
	}
	if !cbMapOK {
		return nil, fmt.Errorf("tensai: gpu readback failed: %s %s", cbFailMsg, uncapturedCB)
	}
	src := fnBufferGetConstMapped(buf, 0, uintptr(bytes))
	if src == nil {
		return nil, errors.New("tensai: gpu readback: mapped range unavailable")
	}
	return src, nil
}
