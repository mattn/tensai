//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu_test

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
)

// The tests in this package drive the device through autograd, which
// imports gpu itself -- so they live outside the package to keep the
// import graph acyclic, with their own copies of the two helpers.

func openTestGPU(t *testing.T) *gpu.Device {
	t.Helper()
	g, err := gpu.Open()
	if err != nil {
		t.Skipf("wgpu unavailable: %v", err)
	}
	return g
}

func randTensor(rng *rand.Rand, shape ...int) *tensai.Tensor {
	x := tensai.NewTensor(shape...)
	for i := range x.Data {
		x.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return x
}
