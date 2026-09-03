//go:build !goexperiment.simd || (!amd64 && !arm64)

package quant

import "github.com/mattn/tensai"

// Portable dispatcher for the activation quantizer; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quantacts_simd.go.

func quantizeActsInto(x []tensai.Float, xu []uint8) tensai.Float {
	return quantizeActsScalar(x, xu)
}
