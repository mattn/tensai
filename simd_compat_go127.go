//go:build goexperiment.simd && amd64 && go1.27

package tensai

import "simd/archsimd"

// Thin wrappers over the archsimd calls whose names changed between Go
// releases; the experimental package is not covered by the compatibility
// promise. This file targets the Go 1.27 API, simd_compat_go126.go the
// older one. The kernels in dot_simd.go and mathvec_simd.go use only these
// names for loads, stores, and rounding.

func loadF32x8(s []Float) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8(s)
}

func loadF32x8Part(s []Float) archsimd.Float32x8 {
	v, _ := archsimd.LoadFloat32x8Part(s)
	return v
}

func storeF32x8(v archsimd.Float32x8, s []Float) {
	v.Store(s)
}

func storeF32x8Part(v archsimd.Float32x8, s []Float) {
	v.StorePart(s)
}

func roundEven(v archsimd.Float32x8) archsimd.Float32x8 {
	return v.Round()
}
