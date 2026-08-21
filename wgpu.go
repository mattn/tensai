//go:build wgpu && (linux || darwin)

package tensai

// A WebGPU compute backend for batched MatMul, talking to the wgpu-native
// shared library (https://github.com/gfx-rs/wgpu-native) through purego —
// no cgo, no C compiler, just dlopen at runtime. wgpu-native routes to
// Vulkan, Metal, or D3D12, so this covers AMD, Intel, Apple, and NVIDIA
// GPUs (and CPU fallbacks like lavapipe) with one implementation.
//
// Build with -tags wgpu and put libwgpu_native.so / .dylib where the
// dynamic loader finds it, or point TENSAI_WGPU_LIB at it. The bindings
// target the wgpu-native v22.1.0.5 C API (the last release before the
// callback-info API rework); use that release's binaries.
//
// Only OpenGPU and the GPU methods are exported; everything else mirrors
// webgpu.h struct layouts by hand, which is why this file is long.

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
	mu       sync.Mutex
	instance uintptr
	adapter  uintptr
	device   uintptr
	queue    uintptr
	module   uintptr
	pipeline uintptr
	layout   uintptr // bind group layout, owned by the pipeline
	name     string
	closed   bool
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

// webgpu.h constants (v22.1.0.5).
const (
	wgpuSTypeShaderModuleWGSLDescriptor = 0x00000006

	wgpuBufferUsageMapRead = 0x00000001
	wgpuBufferUsageCopySrc = 0x00000004
	wgpuBufferUsageCopyDst = 0x00000008
	wgpuBufferUsageUniform = 0x00000040
	wgpuBufferUsageStorage = 0x00000080

	wgpuMapModeRead = 0x00000001

	wgpuStatusSuccess = 0 // shared by request-adapter/device and map-async
)

// Hand-mirrored webgpu.h structs. Field order, widths, and padding must
// match the 64-bit C layouts exactly.

type wgpuChainedStruct struct {
	next  uintptr
	sType uint32
	_     uint32
}

type wgpuShaderModuleWGSLDescriptor struct {
	chain wgpuChainedStruct
	code  *byte
}

type wgpuShaderModuleDescriptor struct {
	nextInChain *wgpuShaderModuleWGSLDescriptor
	label       uintptr
	hintCount   uintptr
	hints       uintptr
}

type wgpuProgrammableStageDescriptor struct {
	nextInChain   uintptr
	module        uintptr
	entryPoint    *byte
	constantCount uintptr
	constants     uintptr
}

type wgpuComputePipelineDescriptor struct {
	nextInChain uintptr
	label       uintptr
	layout      uintptr // 0 = auto layout
	compute     wgpuProgrammableStageDescriptor
}

type wgpuBufferDescriptor struct {
	nextInChain      uintptr
	label            uintptr
	usage            uint32
	_                uint32
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
	label       uintptr
	layout      uintptr
	entryCount  uintptr
	entries     *wgpuBindGroupEntry
}

type wgpuRequestAdapterOptions struct {
	nextInChain       uintptr
	compatibleSurface uintptr
	powerPreference   uint32
	backendType       uint32
	forceFallback     uint32
	_                 uint32
}

type wgpuAdapterInfo struct {
	nextInChain  uintptr
	vendor       uintptr
	architecture uintptr
	device       uintptr
	description  uintptr
	backendType  uint32
	adapterType  uint32
	vendorID     uint32
	deviceID     uint32
}

type wgpuDeviceDescriptor struct {
	nextInChain          uintptr
	label                uintptr
	requiredFeatureCount uintptr
	requiredFeatures     uintptr
	requiredLimits       uintptr
	defaultQueueNext     uintptr // WGPUQueueDescriptor, inlined
	defaultQueueLabel    uintptr
	deviceLostCallback   uintptr
	deviceLostUserdata   uintptr
	errNext              uintptr // WGPUUncapturedErrorCallbackInfo, inlined
	errCallback          uintptr
	errUserdata          uintptr
}

// Function pointers resolved from the shared library.
var (
	fnCreateInstance          func(unsafe.Pointer) uintptr
	fnInstanceRequestAdapter  func(uintptr, unsafe.Pointer, uintptr, unsafe.Pointer)
	fnAdapterGetInfo          func(uintptr, unsafe.Pointer)
	fnAdapterRequestDevice    func(uintptr, unsafe.Pointer, uintptr, unsafe.Pointer)
	fnDeviceGetQueue          func(uintptr) uintptr
	fnDeviceCreateShaderMod   func(uintptr, unsafe.Pointer) uintptr
	fnDeviceCreatePipeline    func(uintptr, unsafe.Pointer) uintptr
	fnPipelineGetLayout       func(uintptr, uint32) uintptr
	fnDeviceCreateBuffer      func(uintptr, unsafe.Pointer) uintptr
	fnDeviceCreateBindGroup   func(uintptr, unsafe.Pointer) uintptr
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
	fnBufferMapAsync          func(uintptr, uint32, uintptr, uintptr, uintptr, unsafe.Pointer)
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

// Callback plumbing. NewCallback slots are permanent, so they are created
// once and route their results through package globals; wgpuMu serializes
// every wgpu call sequence that uses them.
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

func cstr(s string) *byte {
	b := append([]byte(s), 0)
	return &b[0]
}

func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	// p is a C pointer, not a Go object; launder it past the unsafe rules.
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	var out []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(ptr, i))
		if c == 0 {
			return string(out)
		}
		out = append(out, c)
	}
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
			lib, err = purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err == nil {
				break
			}
		}
		if lib == 0 {
			loadErr = fmt.Errorf("tensai: cannot load wgpu-native (set TENSAI_WGPU_LIB): %w", err)
			return
		}
		for _, f := range []struct {
			ptr  any
			name string
		}{
			{&fnCreateInstance, "wgpuCreateInstance"},
			{&fnInstanceRequestAdapter, "wgpuInstanceRequestAdapter"},
			{&fnAdapterGetInfo, "wgpuAdapterGetInfo"},
			{&fnAdapterRequestDevice, "wgpuAdapterRequestDevice"},
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
			{&fnBufferMapAsync, "wgpuBufferMapAsync"},
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

		adapterCB = purego.NewCallback(func(status uint32, adapter, message, userdata uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbAdapter = adapter
			} else {
				cbFailMsg = goString(message)
			}
			return 0
		})
		deviceCB = purego.NewCallback(func(status uint32, device, message, userdata uintptr) uintptr {
			if status == wgpuStatusSuccess {
				cbDevice = device
			} else {
				cbFailMsg = goString(message)
			}
			return 0
		})
		mapCB = purego.NewCallback(func(status uint32, userdata uintptr) uintptr {
			cbMapDone = true
			cbMapOK = status == wgpuStatusSuccess
			return 0
		})
		errorCB = purego.NewCallback(func(errType uint32, message, userdata uintptr) uintptr {
			if uncapturedCB == "" {
				uncapturedCB = goString(message)
			}
			return 0
		})
	})
	return loadErr
}

// matmulWGSL multiplies one (m x k) x (k x n) pair per z-slice; per-batch
// input offsets come from a lookup table so broadcast batches share data.
const matmulWGSL = `
struct Params { m: u32, k: u32, n: u32, batches: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> offs: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let batch = gid.z;
    if (col >= p.n || row >= p.m || batch >= p.batches) { return; }
    let offA = offs[batch].x;
    let offB = offs[batch].y;
    var sum = 0.0;
    for (var i = 0u; i < p.k; i = i + 1u) {
        sum = sum + a[offA + row * p.k + i] * b[offB + i * p.n + col];
    }
    outv[(batch * p.m + row) * p.n + col] = sum;
}
`

// OpenGPU loads wgpu-native, picks an adapter, and compiles the matmul
// pipeline. On machines with several GPUs an optional GPUPower steers the
// choice: GPULowPower prefers the integrated GPU, GPUHighPerformance the
// discrete one. The returned GPU is safe for concurrent use.
func OpenGPU(power ...GPUPower) (*GPU, error) {
	if err := loadWGPU(); err != nil {
		return nil, err
	}
	wgpuMu.Lock()
	defer wgpuMu.Unlock()

	g := &GPU{}
	g.instance = fnCreateInstance(nil)
	if g.instance == 0 {
		return nil, errors.New("tensai: wgpuCreateInstance failed")
	}
	opts := wgpuRequestAdapterOptions{}
	if len(power) > 0 {
		opts.powerPreference = uint32(power[0])
	}
	// wgpu-native invokes these callbacks synchronously, before returning.
	cbAdapter, cbDevice, cbFailMsg = 0, 0, ""
	fnInstanceRequestAdapter(g.instance, unsafe.Pointer(&opts), adapterCB, nil)
	runtime.KeepAlive(&opts)
	if cbAdapter == 0 {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu adapter request failed: %s", cbFailMsg)
	}
	g.adapter = cbAdapter

	// The info strings are tiny and OpenGPU runs once per process, so they
	// are copied and never handed back to wgpuAdapterInfoFreeMembers (which
	// takes the struct by value — off-limits without cgo).
	var info wgpuAdapterInfo
	fnAdapterGetInfo(g.adapter, unsafe.Pointer(&info))
	runtime.KeepAlive(&info)
	g.name = goString(info.device)
	if g.name == "" {
		g.name = goString(info.description)
	}
	switch info.adapterType {
	case 0:
		g.name += " (discrete)"
	case 1:
		g.name += " (integrated)"
	case 2:
		g.name += " (cpu)"
	}

	desc := wgpuDeviceDescriptor{errCallback: errorCB}
	fnAdapterRequestDevice(g.adapter, unsafe.Pointer(&desc), deviceCB, nil)
	runtime.KeepAlive(&desc)
	if cbDevice == 0 {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu device request failed: %s", cbFailMsg)
	}
	g.device = cbDevice
	g.queue = fnDeviceGetQueue(g.device)

	uncapturedCB = ""
	code := cstr(matmulWGSL)
	wgsl := wgpuShaderModuleWGSLDescriptor{
		chain: wgpuChainedStruct{sType: wgpuSTypeShaderModuleWGSLDescriptor},
		code:  code,
	}
	smDesc := wgpuShaderModuleDescriptor{nextInChain: &wgsl}
	g.module = fnDeviceCreateShaderMod(g.device, unsafe.Pointer(&smDesc))
	entry := cstr("main")
	pDesc := wgpuComputePipelineDescriptor{
		compute: wgpuProgrammableStageDescriptor{module: g.module, entryPoint: entry},
	}
	g.pipeline = fnDeviceCreatePipeline(g.device, unsafe.Pointer(&pDesc))
	runtime.KeepAlive(&wgsl)
	runtime.KeepAlive(&smDesc)
	runtime.KeepAlive(&pDesc)
	runtime.KeepAlive(code)
	runtime.KeepAlive(entry)
	if g.pipeline == 0 || uncapturedCB != "" {
		g.Close()
		return nil, fmt.Errorf("tensai: wgpu pipeline creation failed: %s", uncapturedCB)
	}
	g.layout = fnPipelineGetLayout(g.pipeline, 0)
	return g, nil
}

// Name reports the adapter wgpu selected, e.g. "AMD Radeon 780M
// (integrated)" — handy for checking which GPU a power preference landed
// on.
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
	for _, r := range []struct {
		fn func(uintptr)
		h  uintptr
	}{
		{fnLayoutRelease, g.layout},
		{fnPipelineRelease, g.pipeline},
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

func (g *GPU) newBuffer(usage uint32, size uint64) uintptr {
	desc := wgpuBufferDescriptor{usage: usage, size: size}
	buf := fnDeviceCreateBuffer(g.device, unsafe.Pointer(&desc))
	runtime.KeepAlive(&desc)
	return buf
}

// MatMul is the GPU version of the package-level MatMul: identical shape
// and broadcasting semantics, executed as one compute dispatch. Inputs and
// the result live in host memory; each call uploads the operands and reads
// the product back, so it only pays off for large products.
func (g *GPU) MatMul(a, b *Tensor) (*Tensor, error) {
	na, nb := len(a.Shape), len(b.Shape)
	if na < 2 || nb < 2 {
		return nil, fmt.Errorf("tensai: matmul needs at least 2 axes: %v * %v", a.Shape, b.Shape)
	}
	m, k := a.Shape[na-2], a.Shape[na-1]
	if b.Shape[nb-2] != k {
		return nil, fmt.Errorf("tensai: matmul shape mismatch: %v * %v", a.Shape, b.Shape)
	}
	n := b.Shape[nb-1]
	batch, err := broadcastShapes(a.Shape[:na-2], b.Shape[:nb-2])
	if err != nil {
		return nil, err
	}
	out := NewTensor(append(append([]int(nil), batch...), m, n)...)
	batches := prodDims(batch)
	if batches > 65535 {
		return nil, fmt.Errorf("tensai: gpu matmul batch count %d exceeds 65535", batches)
	}

	// Element offsets of each batch's matrix in a and b.
	as := broadcastStrides(a.Shape[:na-2], batch)
	bs := broadcastStrides(b.Shape[:nb-2], batch)
	offs := make([]uint32, 2*batches)
	for bi := 0; bi < batches; bi++ {
		offA, offB := 0, 0
		for d, rem := len(batch)-1, bi; d >= 0; d-- {
			i := rem % batch[d]
			rem /= batch[d]
			offA += i * as[d]
			offB += i * bs[d]
		}
		offs[2*bi] = uint32(offA * m * k)
		offs[2*bi+1] = uint32(offB * k * n)
	}
	params := [4]uint32{uint32(m), uint32(k), uint32(n), uint32(batches)}

	g.mu.Lock()
	defer g.mu.Unlock()
	wgpuMu.Lock()
	defer wgpuMu.Unlock()
	if g.closed {
		return nil, errors.New("tensai: gpu is closed")
	}
	uncapturedCB = ""

	outBytes := uint64(len(out.Data)) * 4
	bufParams := g.newBuffer(wgpuBufferUsageUniform|wgpuBufferUsageCopyDst, 16)
	bufA := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(a.Data))*4)
	bufB := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(b.Data))*4)
	bufOffs := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopyDst, uint64(len(offs))*4)
	bufOut := g.newBuffer(wgpuBufferUsageStorage|wgpuBufferUsageCopySrc, outBytes)
	bufRead := g.newBuffer(wgpuBufferUsageMapRead|wgpuBufferUsageCopyDst, outBytes)
	defer func() {
		for _, buf := range []uintptr{bufParams, bufA, bufB, bufOffs, bufOut, bufRead} {
			if buf != 0 {
				fnBufferRelease(buf)
			}
		}
	}()

	fnQueueWriteBuffer(g.queue, bufParams, 0, unsafe.Pointer(&params[0]), 16)
	fnQueueWriteBuffer(g.queue, bufA, 0, unsafe.Pointer(&a.Data[0]), uintptr(len(a.Data))*4)
	fnQueueWriteBuffer(g.queue, bufB, 0, unsafe.Pointer(&b.Data[0]), uintptr(len(b.Data))*4)
	fnQueueWriteBuffer(g.queue, bufOffs, 0, unsafe.Pointer(&offs[0]), uintptr(len(offs))*4)

	entries := [5]wgpuBindGroupEntry{
		{binding: 0, buffer: bufParams, size: 16},
		{binding: 1, buffer: bufA, size: uint64(len(a.Data)) * 4},
		{binding: 2, buffer: bufB, size: uint64(len(b.Data)) * 4},
		{binding: 3, buffer: bufOffs, size: uint64(len(offs)) * 4},
		{binding: 4, buffer: bufOut, size: outBytes},
	}
	bgDesc := wgpuBindGroupDescriptor{layout: g.layout, entryCount: 5, entries: &entries[0]}
	bindGroup := fnDeviceCreateBindGroup(g.device, unsafe.Pointer(&bgDesc))
	runtime.KeepAlive(&entries)
	runtime.KeepAlive(&bgDesc)
	defer fnBindGroupRelease(bindGroup)

	encoder := fnDeviceCreateCmdEncoder(g.device, nil)
	pass := fnEncoderBeginComputePass(encoder, nil)
	fnPassSetPipeline(pass, g.pipeline)
	fnPassSetBindGroup(pass, 0, bindGroup, 0, nil)
	fnPassDispatch(pass, uint32((n+7)/8), uint32((m+7)/8), uint32(batches))
	fnPassEnd(pass)
	fnPassRelease(pass)
	fnEncoderCopyBuffer(encoder, bufOut, 0, bufRead, 0, outBytes)
	cmd := fnEncoderFinish(encoder, nil)
	fnCmdEncoderRelease(encoder)
	fnQueueSubmit(g.queue, 1, unsafe.Pointer(&cmd))
	fnCmdBufferRelease(cmd)

	cbMapDone, cbMapOK = false, false
	fnBufferMapAsync(bufRead, wgpuMapModeRead, 0, uintptr(outBytes), mapCB, nil)
	for !cbMapDone {
		fnDevicePoll(g.device, 1, nil)
	}
	if !cbMapOK {
		return nil, fmt.Errorf("tensai: gpu matmul readback failed: %s", uncapturedCB)
	}
	src := fnBufferGetConstMapped(bufRead, 0, uintptr(outBytes))
	if src == nil {
		return nil, errors.New("tensai: gpu matmul: mapped range unavailable")
	}
	copy(out.Data, unsafe.Slice((*Float)(src), len(out.Data)))
	fnBufferUnmap(bufRead)
	if uncapturedCB != "" {
		return nil, fmt.Errorf("tensai: gpu matmul failed: %s", uncapturedCB)
	}
	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	return out, nil
}
