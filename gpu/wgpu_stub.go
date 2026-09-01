//go:build !((wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows)))

package gpu

import (
	"errors"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/quant"
)

// Device is the WebGPU compute backend. This build has it disabled; build with
// -tags wgpu (linux, darwin, or windows) and see wgpu.go for the runtime
// requirements.
type Device struct{}

// Power tells Open which adapter to prefer; unused in builds without
// the wgpu tag.
type Power uint32

const (
	Default         Power = 0
	LowPower        Power = 1
	HighPerformance Power = 2
)

var errNoWGPU = errors.New("tensai: built without wgpu support (rebuild with -tags wgpu)")

// Open always fails in builds without the wgpu tag.
func Open(power ...Power) (*Device, error) { return nil, errNoWGPU }

// Backend returns the empty string: this build has no GPU support.
func Backend() string { return "" }

// Name always returns "" in builds without the wgpu tag.
func (g *Device) Name() string { return "" }

// MatMul always fails in builds without the wgpu tag.
func (g *Device) MatMul(a, b *tensai.Tensor) (*tensai.Tensor, error) { return nil, errNoWGPU }

// MatMulTN is a no-op stub; build with -tags wgpu or wgpu24.
func (g *Device) MatMulTN(a, b *tensai.Tensor) (*tensai.Tensor, error) { return nil, errNoWGPU }

// MatMulNT is a no-op stub; build with -tags wgpu or wgpu24.
func (g *Device) MatMulNT(a, b *tensai.Tensor) (*tensai.Tensor, error) { return nil, errNoWGPU }

// Close is a no-op in builds without the wgpu tag.
func (g *Device) Close() {}

// StorageLimit returns 0 in builds without the wgpu tag.
func (g *Device) StorageLimit() uint64 { return 0 }

// Tensor is a Device-resident tensor. This build has it disabled; build
// with -tags wgpu or -tags wgpu24 to enable it.
type Tensor struct{}

// Upload always fails in builds without the wgpu tag.
func (g *Device) Upload(t *tensai.Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMul always fails in builds without the wgpu tag.
func (t *Tensor) MatMul(o *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulT always fails in builds without the wgpu tag.
func (t *Tensor) MatMulT(o *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulTN is a no-op stub; build with -tags wgpu or wgpu24.
func (t *Tensor) MatMulTN(o *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// Scale always fails in builds without the wgpu tag.
func (t *Tensor) Scale(s tensai.Float) error { return errNoWGPU }

// Softmax always fails in builds without the wgpu tag.
func (t *Tensor) Softmax() (*Tensor, error) { return nil, errNoWGPU }

// Attention always fails in builds without the wgpu tag.
func (q *Tensor) Attention(k, v *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MultiHeadAttention always fails in builds without the wgpu tag.
func (q *Tensor) MultiHeadAttention(k, v *Tensor, heads int) (*Tensor, error) {
	return nil, errNoWGPU
}

// CausalAttention always fails in builds without the wgpu tag.
func (q *Tensor) CausalAttention(k, v *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// CausalMultiHeadAttention always fails in builds without the wgpu tag.
func (q *Tensor) CausalMultiHeadAttention(k, v *Tensor, heads int) (*Tensor, error) {
	return nil, errNoWGPU
}

// Download always fails in builds without the wgpu tag.
func (t *Tensor) Download() (*tensai.Tensor, error) { return nil, errNoWGPU }

// Free is a no-op in builds without the wgpu tag.
func (t *Tensor) Free() {}

// View always fails in builds without the wgpu tag.
func (t *Tensor) View(off int, shape ...int) (*Tensor, error) { return nil, errNoWGPU }

// SliceCols fails without a GPU build.
func (t *Tensor) SliceCols(off, cols int) (*Tensor, error) { return nil, errNoWGPU }

// Shape returns nil in builds without the wgpu tag.
func (t *Tensor) Shape() []int { return nil }

// Size returns 0 in builds without the wgpu tag.
func (t *Tensor) Size() int { return 0 }

// RMSNorm always fails in builds without the wgpu tag.
func (t *Tensor) RMSNorm(w *Tensor, eps float64) (*Tensor, error) { return nil, errNoWGPU }

// RoPE always fails in builds without the wgpu tag.
func (t *Tensor) RoPE(headSz, pos0 int, theta float64) error { return errNoWGPU }

// Add always fails in builds without the wgpu tag.
func (t *Tensor) Add(o *Tensor) error { return errNoWGPU }

// SiluMul always fails in builds without the wgpu tag.
func (t *Tensor) SiluMul(o *Tensor) error { return errNoWGPU }

// GLUSplit always fails in builds without the wgpu tag.
func (t *Tensor) GLUSplit(inter int, gelu bool) (*Tensor, error) { return nil, errNoWGPU }

// CopyRowsInto always fails in builds without the wgpu tag.
func (t *Tensor) CopyRowsInto(dst *Tensor, off int) error { return errNoWGPU }

// GroupedCausalAttention always fails in builds without the wgpu tag.
// GroupedCausalAttentionParts always fails in builds without the wgpu tag.
func (q *Tensor) GroupedCausalAttentionParts(k, v *Tensor, heads, kvHeads, seqKV, window int) (*Tensor, int, error) {
	return nil, 0, errNoWGPU
}

func (q *Tensor) GroupedCausalAttention(k, v *Tensor, heads, kvHeads, seqKV, window int) (*Tensor, error) {
	return nil, errNoWGPU
}

// GeluMul always fails in builds without the wgpu tag.
func (t *Tensor) GeluMul(o *Tensor) error { return errNoWGPU }

// QMatrix is a Device-resident int8 weight matrix. This build has it
// disabled; build with -tags wgpu or -tags wgpu24 to enable it.
type QMatrix struct{}

// UploadQ8 always fails in builds without the wgpu tag.
func (g *Device) UploadQ8(q *quant.QMatrix) (*QMatrix, error) { return nil, errNoWGPU }

// MatMul always fails in builds without the wgpu tag.
func (q *QMatrix) MatMul(x *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulOpts always fails in builds without the wgpu tag.
func (q *QMatrix) MatMulOpts(x, bias, dst *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulRMSNorm always fails in builds without the wgpu tag.
func (q *QMatrix) MatMulRMSNorm(x, norm *Tensor, eps float64, bias, dst *Tensor) (*Tensor, error) {
	return nil, errNoWGPU
}

// MatMulAttnCombine always fails in builds without the wgpu tag.
func (q *QMatrix) MatMulAttnCombine(scr *Tensor, slabs, dh, group int, bias, dst *Tensor) (*Tensor, error) {
	return nil, errNoWGPU
}

// Shape returns zeros in builds without the wgpu tag.
func (q *QMatrix) Shape() (int, int) { return 0, 0 }

// Free is a no-op in builds without the wgpu tag.
func (q *QMatrix) Free() {}

// IsF16 is always false in builds without the wgpu tag.
func (t *Tensor) IsF16() bool { return false }

// RopeCacheF16 always fails in builds without the wgpu tag.
func (t *Tensor) RopeCacheF16(kc, vc *Tensor, headSz, qw, kvDim, pos int, theta float64, dstOff int) error {
	return errNoWGPU
}

// BeginBatch always fails in builds without the wgpu tag.
func (g *Device) BeginBatch() error { return errNoWGPU }

// Flush is a no-op in builds without the wgpu tag.
func (g *Device) Flush() error { return nil }

// DownloadRange always fails in builds without the wgpu tag.
func (t *Tensor) DownloadRange(off, n int) (*tensai.Tensor, error) { return nil, errNoWGPU }

// Q4Matrix is a Device-resident int4 weight matrix. This build has it
// disabled; build with -tags wgpu or -tags wgpu24 to enable it.
type Q4Matrix struct{}

// UploadQ4 always fails in builds without the wgpu tag.
func (g *Device) UploadQ4(q *quant.Q4Matrix) (*Q4Matrix, error) { return nil, errNoWGPU }

// HasF16 reports false in builds without the wgpu tag.
func (g *Device) HasF16() bool { return false }

// IntDot always reports false in builds without the wgpu tag.
func (g *Device) IntDot() bool { return false }

// NewF16Tensor always fails in builds without the wgpu tag.
func (g *Device) NewF16Tensor(shape ...int) (*Tensor, error) { return nil, errNoWGPU }

// MatMul always fails in builds without the wgpu tag.
func (q *Q4Matrix) MatMul(x *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulOpts always fails in builds without the wgpu tag.
func (q *Q4Matrix) MatMulOpts(x, bias, dst *Tensor) (*Tensor, error) { return nil, errNoWGPU }

// MatMulRMSNorm always fails in builds without the wgpu tag.
func (q *Q4Matrix) MatMulRMSNorm(x, norm *Tensor, eps float64, bias, dst *Tensor) (*Tensor, error) {
	return nil, errNoWGPU
}

// Shape returns zeros in builds without the wgpu tag.
func (q *Q4Matrix) Shape() (int, int) { return 0, 0 }

// Free is a no-op in builds without the wgpu tag.
func (q *Q4Matrix) Free() {}

// RMSNormEach always fails in builds without the wgpu tag.
func (t *Tensor) RMSNormEach(w *Tensor, eps float64) (*Tensor, error) { return nil, errNoWGPU }
