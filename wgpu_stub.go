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

// GPUTensor is a GPU-resident tensor. This build has it disabled; build
// with -tags wgpu or -tags wgpu24 to enable it.
type GPUTensor struct{}

// Upload always fails in builds without the wgpu tag.
func (g *GPU) Upload(t *Tensor) (*GPUTensor, error) { return nil, errNoWGPU }

// MatMul always fails in builds without the wgpu tag.
func (t *GPUTensor) MatMul(o *GPUTensor) (*GPUTensor, error) { return nil, errNoWGPU }

// MatMulT always fails in builds without the wgpu tag.
func (t *GPUTensor) MatMulT(o *GPUTensor) (*GPUTensor, error) { return nil, errNoWGPU }

// Scale always fails in builds without the wgpu tag.
func (t *GPUTensor) Scale(s Float) error { return errNoWGPU }

// Softmax always fails in builds without the wgpu tag.
func (t *GPUTensor) Softmax() (*GPUTensor, error) { return nil, errNoWGPU }

// Attention always fails in builds without the wgpu tag.
func (q *GPUTensor) Attention(k, v *GPUTensor) (*GPUTensor, error) { return nil, errNoWGPU }

// MultiHeadAttention always fails in builds without the wgpu tag.
func (q *GPUTensor) MultiHeadAttention(k, v *GPUTensor, heads int) (*GPUTensor, error) {
	return nil, errNoWGPU
}

// Download always fails in builds without the wgpu tag.
func (t *GPUTensor) Download() (*Tensor, error) { return nil, errNoWGPU }

// Free is a no-op in builds without the wgpu tag.
func (t *GPUTensor) Free() {}

// Shape returns nil in builds without the wgpu tag.
func (t *GPUTensor) Shape() []int { return nil }

// Size returns 0 in builds without the wgpu tag.
func (t *GPUTensor) Size() int { return 0 }
