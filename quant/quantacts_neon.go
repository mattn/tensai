//go:build goexperiment.simd && arm64

package quant

import "github.com/mattn/tensai"

// The activation quantizer keeps the scalar body on arm64. It is one pass
// over a row, tens of microseconds against a token's tens of milliseconds,
// and its rounding leans on an int-to-float bit cast that NEON's exposed
// API does not have.
func quantizeActsInto(x []tensai.Float, xu []uint8) tensai.Float {
	return quantizeActsScalar(x, xu)
}
