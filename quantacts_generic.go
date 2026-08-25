//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatcher for the activation quantizer; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quantacts_simd.go.

func quantizeActsInto(x []Float, xu []uint8) Float {
	return quantizeActsScalar(x, xu)
}
