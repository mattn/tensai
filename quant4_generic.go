//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatcher for the 4-bit matvec kernel; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 version in quant4_simd.go.

func q4matvecCols(out, x []Float, qw []byte, scale []Float, cols, lo, hi int, tmp []Float) {
	q4matvecColsGeneric(out, x, qw, scale, cols, lo, hi, tmp)
}
