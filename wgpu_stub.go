//go:build !((wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin)))

package tensai

import "errors"

// GPU is the WebGPU compute backend. This build has it disabled; build with
// -tags wgpu (linux, darwin, or windows) and see wgpu.go for the runtime
// requirements.
type GPU struct{}

// GPUPower tells OpenGPU which adapter to prefer; unused in builds without
// the wgpu tag.
type GPUPower uint32

const (
	GPUDefault         GPUPower = 0
	GPULowPower        GPUPower = 1
	GPUHighPerformance GPUPower = 2
)

var errNoWGPU = errors.New("tensai: built without wgpu support (rebuild with -tags wgpu)")

// OpenGPU always fails in builds without the wgpu tag.
func OpenGPU(power ...GPUPower) (*GPU, error) { return nil, errNoWGPU }

// Name always returns "" in builds without the wgpu tag.
func (g *GPU) Name() string { return "" }

// MatMul always fails in builds without the wgpu tag.
func (g *GPU) MatMul(a, b *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// Close is a no-op in builds without the wgpu tag.
func (g *GPU) Close() {}
