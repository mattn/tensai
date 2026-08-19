//go:build !goexperiment.simd || !amd64

package tensai

// dotRows computes rows lo..hi of out = a * b. This is the portable
// fallback; build with GOEXPERIMENT=simd on amd64 to get the AVX2 kernel in
// dot_simd.go.
func dotRows(out, a, b *Matrix, lo, hi int) {
	dotRowsGeneric(out, a, b, lo, hi)
}
