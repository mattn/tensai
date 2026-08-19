//go:build goexperiment.simd && amd64

package tensai

import (
	"simd/archsimd"
	"unsafe"
)

// hasAVX2 gates the vector kernel on the CPU actually supporting AVX2+FMA.
var hasAVX2 = archsimd.X86.AVX2() && archsimd.X86.FMA()

// dotRows computes rows lo..hi of out = a * b with 8-lane float32 fused
// multiply-adds, unrolled 4x.
//
// Scalar float instructions use SSE encodings, and interleaving those with
// 256-bit AVX while the upper vector state is dirty stalls the pipeline, so
// the k loop must stay free of them: the zero test reads the float's bits
// as an integer, the broadcast goes through an integer register
// (VPBROADCASTD) with a free bit-cast, and the column tail uses masked
// part-loads instead of a scalar loop.
func dotRows(out, a, b *Matrix, lo, hi int) {
	if !hasAVX2 {
		dotRowsGeneric(out, a, b, lo, hi)
		return
	}
	// Leave the vector unit's upper state clean for surrounding SSE code.
	defer archsimd.ClearAVXUpperBits()

	cols := b.Cols
	wide := cols &^ 31 // widest multiple of 32
	vecs := cols &^ 7  // widest multiple of 8
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		aBits := unsafe.Slice((*uint32)(unsafe.Pointer(&aRow[0])), len(aRow))
		outRow := out.Data[r*cols : (r+1)*cols]
		for k := range aBits {
			if aBits[k]<<1 == 0 { // +0.0 or -0.0
				continue
			}
			bRow := b.Data[k*cols : (k+1)*cols]
			vv := archsimd.BroadcastUint32x8(aBits[k]).AsFloat32x8()
			var c int
			for ; c < wide; c += 32 {
				archsimd.LoadFloat32x8Slice(bRow[c:]).
					MulAdd(vv, archsimd.LoadFloat32x8Slice(outRow[c:])).StoreSlice(outRow[c:])
				archsimd.LoadFloat32x8Slice(bRow[c+8:]).
					MulAdd(vv, archsimd.LoadFloat32x8Slice(outRow[c+8:])).StoreSlice(outRow[c+8:])
				archsimd.LoadFloat32x8Slice(bRow[c+16:]).
					MulAdd(vv, archsimd.LoadFloat32x8Slice(outRow[c+16:])).StoreSlice(outRow[c+16:])
				archsimd.LoadFloat32x8Slice(bRow[c+24:]).
					MulAdd(vv, archsimd.LoadFloat32x8Slice(outRow[c+24:])).StoreSlice(outRow[c+24:])
			}
			for ; c < vecs; c += 8 {
				archsimd.LoadFloat32x8Slice(bRow[c:]).
					MulAdd(vv, archsimd.LoadFloat32x8Slice(outRow[c:])).StoreSlice(outRow[c:])
			}
			if c < cols {
				archsimd.LoadFloat32x8SlicePart(bRow[c:]).
					MulAdd(vv, archsimd.LoadFloat32x8SlicePart(outRow[c:])).StoreSlicePart(outRow[c:])
			}
		}
	}
}
