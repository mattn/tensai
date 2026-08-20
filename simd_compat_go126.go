//go:build goexperiment.simd && amd64 && !go1.27

package tensai

import "simd/archsimd"

// Go 1.26 spellings of the archsimd calls whose names changed in Go 1.27;
// see simd_compat_go127.go.

func loadF32x8(s []Float) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Slice(s)
}

func loadF32x8Part(s []Float) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8SlicePart(s)
}

func storeF32x8(v archsimd.Float32x8, s []Float) {
	v.StoreSlice(s)
}

func storeF32x8Part(v archsimd.Float32x8, s []Float) {
	v.StoreSlicePart(s)
}

func roundEven(v archsimd.Float32x8) archsimd.Float32x8 {
	return v.RoundToEven()
}
