//go:build goexperiment.simd && amd64 && go1.27

package simd

import "simd/archsimd"

// Thin wrappers over the archsimd calls whose names changed between Go
// releases; the experimental package is not covered by the compatibility
// promise. This file targets the Go 1.27 API, compat_go126.go the
// older one. The kernels in dot_simd.go and mathvec_simd.go use only these
// names for loads, stores, and rounding.

func LoadF32x8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8(s)
}

func LoadF32x8Part(s []float32) archsimd.Float32x8 {
	v, _ := archsimd.LoadFloat32x8Part(s)
	return v
}

func StoreF32x8(v archsimd.Float32x8, s []float32) {
	v.Store(s)
}

func LoadI32x8(s []int32) archsimd.Int32x8 {
	return archsimd.LoadInt32x8(s)
}

func StoreI32x8(v archsimd.Int32x8, s []int32) {
	v.Store(s)
}

func LoadU32x8(s []uint32) archsimd.Uint32x8 {
	return archsimd.LoadUint32x8(s)
}

func LoadU8x16(s []uint8) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16(s)
}

func LoadI8x16(s []int8) archsimd.Int8x16 {
	return archsimd.LoadInt8x16(s)
}

func LoadI8x32(s []int8) archsimd.Int8x32 {
	return archsimd.LoadInt8x32(s)
}

func StoreF32x8Part(v archsimd.Float32x8, s []float32) {
	v.StorePart(s)
}

func RoundEven(v archsimd.Float32x8) archsimd.Float32x8 {
	return v.Round()
}
