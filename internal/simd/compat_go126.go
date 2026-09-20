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

func LoadU8x32(s []uint8) archsimd.Uint8x32 {
	return archsimd.LoadUint8x32Slice(s)
}

// LoadF32x8Part loads up to eight floats, zero-filling the rest, through
// an array rather than archsimd's part load; see compat_go127.go for why.
func LoadF32x8Part(s []float32) archsimd.Float32x8 {
	if len(s) >= 8 {
		return archsimd.LoadFloat32x8(s)
	}
	var buf [8]float32
	copy(buf[:], s)
	return archsimd.LoadFloat32x8(buf[:])
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

// MulSignI8x32 multiplies x by the sign of y (VPSIGNB); Go 1.26 spells
// the method CopySign.
func MulSignI8x32(x, y archsimd.Int8x32) archsimd.Int8x32 { return x.CopySign(y) }
