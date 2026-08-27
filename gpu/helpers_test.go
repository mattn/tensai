//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"math/rand"

	"github.com/mattn/tensai"
)

func randTensor(rng *rand.Rand, shape ...int) *tensai.Tensor {
	x := tensai.NewTensor(shape...)
	for i := range x.Data {
		x.Data[i] = tensai.Float(rng.NormFloat64())
	}
	return x
}
