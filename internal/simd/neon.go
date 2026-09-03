//go:build goexperiment.simd && arm64 && go1.27

package simd

import "simd/archsimd"

// The arm64 half of the load/store wrappers: 128-bit NEON vectors where
// the amd64 files use 256-bit AVX2 ones. Every arm64 chip Go targets has
// NEON, so there is no feature gate to check the way HasAVX2 gates the
// x86 kernels; HasAVX2 stays defined and false so code that asks about
// AVX2 keeps compiling here.
const (
	HasAVX2 = false
	HasNEON = true
)

func LoadF32x4(s []float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4(s) }

func StoreF32x4(v archsimd.Float32x4, s []float32) { v.Store(s) }

func LoadI32x4(s []int32) archsimd.Int32x4 { return archsimd.LoadInt32x4(s) }

func StoreI32x4(v archsimd.Int32x4, s []int32) { v.Store(s) }

func LoadI8x16(s []int8) archsimd.Int8x16 { return archsimd.LoadInt8x16(s) }

func LoadU8x16(s []uint8) archsimd.Uint8x16 { return archsimd.LoadUint8x16(s) }

func LoadU32x4(s []uint32) archsimd.Uint32x4 { return archsimd.LoadUint32x4(s) }

func LoadF32x4Part(s []float32) archsimd.Float32x4 {
	v, _ := archsimd.LoadFloat32x4Part(s)
	return v
}

func StoreF32x4Part(v archsimd.Float32x4, s []float32) { v.StorePart(s) }
