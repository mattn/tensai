//go:build goexperiment.simd && amd64

package simd

import "simd/archsimd"

// HasAVX2 gates the vector kernels on the CPU actually supporting AVX2+FMA.
var HasAVX2 = archsimd.X86.AVX2() && archsimd.X86.FMA()
