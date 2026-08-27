//go:build goexperiment.simd && amd64 && !go1.27

package simd

import "simd/archsimd"

// Go 1.26 spellings of the archsimd calls whose names changed in Go 1.27;
// see compat_go127.go.

func LoadF32x8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Slice(s)
}

func LoadI32x8(s []int32) archsimd.Int32x8 {
	return archsimd.LoadInt32x8Slice(s)
}

func StoreI32x8(v archsimd.Int32x8, s []int32) {
	v.StoreSlice(s)
}

func LoadU32x8(s []uint32) archsimd.Uint32x8 {
	return archsimd.LoadUint32x8Slice(s)
}

func LoadU8x16(s []uint8) archsimd.Uint8x16 {
	return archsimd.LoadUint8x16Slice(s)
}

func LoadI8x16(s []int8) archsimd.Int8x16 {
	return archsimd.LoadInt8x16Slice(s)
}

func LoadI8x32(s []int8) archsimd.Int8x32 {
	return archsimd.LoadInt8x32Slice(s)
}

func LoadF32x8Part(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8SlicePart(s)
}

func StoreF32x8(v archsimd.Float32x8, s []float32) {
	v.StoreSlice(s)
}

func StoreF32x8Part(v archsimd.Float32x8, s []float32) {
	v.StoreSlicePart(s)
}

func RoundEven(v archsimd.Float32x8) archsimd.Float32x8 {
	return v.RoundToEven()
}
