package tensai

import "sync/atomic"

// Accelerator is a backend that can run the three stacked products faster
// than the CPU kernels: the forward `a * b`, the input gradient `a * b^T`,
// and the weight gradient `a^T * b`. A gpu.Device implements it, so
//
//	dev, err := gpu.Open(gpu.HighPerformance)
//	tensai.UseAccelerator(dev)
//
// moves every product above the size threshold -- including both halves of
// an autograd backward pass -- onto the GPU, and leaves everything smaller
// on the CPU, where the kernels win.
//
// An accelerator must return a freshly allocated result with the shape
// MatMul, MatMulNT and MatMulTN produce, and must be safe to call from
// several goroutines. An error is never fatal: the product is simply run
// on the CPU instead.
type Accelerator interface {
	MatMul(a, b *Tensor) (*Tensor, error)
	MatMulTN(a, b *Tensor) (*Tensor, error)
	MatMulNT(a, b *Tensor) (*Tensor, error)
}

// accel holds the installed accelerator (an atomic.Value of accelState so a
// product can read it without locking).
var accel atomic.Value

type accelState struct {
	acc Accelerator
	min int64 // multiply-accumulates below which the CPU is used
}

// DefaultAcceleratorThreshold is the multiply-accumulate count above which
// an installed accelerator is used. Below it the round trip through the
// device costs more than the CPU kernels take: on an AMD 780M a 512x512x512
// product (1.3e8) is a small loss and a 1024-cube one (1.1e9) is a 2x win,
// so the default sits between them.
const DefaultAcceleratorThreshold = 4e8

// UseAccelerator installs acc for products at or above the default
// threshold. Passing nil removes it. It returns the previous accelerator.
func UseAccelerator(acc Accelerator) Accelerator {
	return UseAcceleratorThreshold(acc, DefaultAcceleratorThreshold)
}

// UseAcceleratorThreshold installs acc for products of at least minMACs
// multiply-accumulates (m*k*n, times the batch count). A threshold of 0
// sends every product to the accelerator, which is mostly useful in tests.
func UseAcceleratorThreshold(acc Accelerator, minMACs int64) Accelerator {
	prev := Acceleration()
	accel.Store(accelState{acc: acc, min: minMACs})
	return prev
}

// Acceleration returns the installed accelerator, or nil.
func Acceleration() Accelerator {
	s, _ := accel.Load().(accelState)
	return s.acc
}

// acceleratorFor returns the accelerator to run a product of the given size
// on, or nil to stay on the CPU.
func acceleratorFor(macs int64) Accelerator {
	s, _ := accel.Load().(accelState)
	if s.acc == nil || macs < s.min {
		return nil
	}
	return s.acc
}
