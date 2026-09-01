//go:build goexperiment.simd && amd64

package tensai

import (
	"runtime"
	"simd/archsimd"
	"unsafe"

	"github.com/mattn/tensai/internal/simd"
)

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
	if !simd.HasAVX2 {
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
		initialized := false
		for k := range aBits {
			if aBits[k]<<1 == 0 { // +0.0 or -0.0
				continue
			}
			bRow := b.Data[k*cols : (k+1)*cols]
			vv := archsimd.BroadcastUint32x8(aBits[k]).AsFloat32x8()
			var c int
			if !initialized {
				for ; c < wide; c += 32 {
					simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).Mul(vv), outRow[c:])
					simd.StoreF32x8(simd.LoadF32x8(bRow[c+8:]).Mul(vv), outRow[c+8:])
					simd.StoreF32x8(simd.LoadF32x8(bRow[c+16:]).Mul(vv), outRow[c+16:])
					simd.StoreF32x8(simd.LoadF32x8(bRow[c+24:]).Mul(vv), outRow[c+24:])
				}
				for ; c < vecs; c += 8 {
					simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).Mul(vv), outRow[c:])
				}
				if c < cols {
					simd.StoreF32x8Part(simd.LoadF32x8Part(bRow[c:]).Mul(vv), outRow[c:])
				}
				initialized = true
				continue
			}
			for ; c < wide; c += 32 {
				simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).MulAdd(vv, simd.LoadF32x8(outRow[c:])), outRow[c:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+8:]).MulAdd(vv, simd.LoadF32x8(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+16:]).MulAdd(vv, simd.LoadF32x8(outRow[c+16:])), outRow[c+16:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+24:]).MulAdd(vv, simd.LoadF32x8(outRow[c+24:])), outRow[c+24:])
			}
			for ; c < vecs; c += 8 {
				simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).MulAdd(vv, simd.LoadF32x8(outRow[c:])), outRow[c:])
			}
			if c < cols {
				simd.StoreF32x8Part(simd.LoadF32x8Part(bRow[c:]).MulAdd(vv, simd.LoadF32x8Part(outRow[c:])), outRow[c:])
			}
		}
		if !initialized {
			clear(outRow)
		}
	}
}

func dotWorkerCount(rows, inner, cols int) int {
	workers := 1
	if rows*inner*cols >= 1<<23 {
		workers = runtime.NumCPU()
		if workers > rows {
			workers = rows
		}
	}
	return workers
}

// dotTARows computes out rows lo..hi of out = a^T * b with the same
// SSE-free 8-lane FMA pattern as dotRows: a's element is fetched as integer
// bits for the zero test and broadcast through an integer register.
// dotTATall computes out = a^T * b when b has at most eight columns: the
// output rows lo..hi are accumulated four at a time in vector registers,
// so the inputs are streamed instead of the output being read back and
// written for every element. The general kernel below does the opposite,
// which is right when the output is wide and wrong when it is a handful of
// values -- the shape a convolution's weight gradient has, where the
// contracted axis is every pixel of every image in the batch.
func dotTATall(out, a, b *Matrix, lo, hi int) {
	if !simd.HasAVX2 {
		dotTARowsGeneric(out, a, b, lo, hi)
		return
	}
	defer archsimd.ClearAVXUpperBits()

	k, n, rows := a.Cols, b.Cols, a.Rows
	for j0 := 0; j0 < n; j0 += 8 {
		width := min(8, n-j0)
		dotTATallCols(out, a, b, lo, hi, k, n, rows, j0, width)
	}
}

// dotTATallCols is dotTATall over one eight-wide slice of b's columns.
func dotTATallCols(out, a, b *Matrix, lo, hi, k, n, rows, j0, width int) {
	for i0 := lo; i0 < hi; i0 += 4 {
		var acc0, acc1, acc2, acc3 archsimd.Float32x8
		switch hi - i0 {
		case 1:
			for r := 0; r < rows; r++ {
				bv := simd.LoadF32x8Part(b.Data[r*n+j0 : r*n+j0+width])
				aRow := a.Data[r*k+i0:]
				acc0 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[0]), acc0)
			}
		case 2:
			for r := 0; r < rows; r++ {
				bv := simd.LoadF32x8Part(b.Data[r*n+j0 : r*n+j0+width])
				aRow := a.Data[r*k+i0:]
				acc0 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[0]), acc0)
				acc1 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[1]), acc1)
			}
		case 3:
			for r := 0; r < rows; r++ {
				bv := simd.LoadF32x8Part(b.Data[r*n+j0 : r*n+j0+width])
				aRow := a.Data[r*k+i0:]
				acc0 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[0]), acc0)
				acc1 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[1]), acc1)
				acc2 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[2]), acc2)
			}
		default:
			for r := 0; r < rows; r++ {
				bv := simd.LoadF32x8Part(b.Data[r*n+j0 : r*n+j0+width])
				aRow := a.Data[r*k+i0:]
				acc0 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[0]), acc0)
				acc1 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[1]), acc1)
				acc2 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[2]), acc2)
				acc3 = bv.MulAdd(archsimd.BroadcastFloat32x8(aRow[3]), acc3)
			}
		}
		accs := [4]archsimd.Float32x8{acc0, acc1, acc2, acc3}
		for i := i0; i < hi && i < i0+4; i++ {
			simd.StoreF32x8Part(accs[i-i0], out.Data[i*n+j0:i*n+j0+width])
		}
	}
}

func dotTARows(out, a, b *Matrix, lo, hi int) {
	if !simd.HasAVX2 {
		dotTARowsGeneric(out, a, b, lo, hi)
		return
	}
	defer archsimd.ClearAVXUpperBits()

	cols := b.Cols
	wide := cols &^ 31
	vecs := cols &^ 7
	for r := 0; r < a.Rows; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		aBits := unsafe.Slice((*uint32)(unsafe.Pointer(&aRow[0])), len(aRow))
		bRow := b.Data[r*cols : (r+1)*cols]
		for i := lo; i < hi; i++ {
			if aBits[i]<<1 == 0 { // +0.0 or -0.0
				continue
			}
			outRow := out.Data[i*cols : (i+1)*cols]
			vv := archsimd.BroadcastUint32x8(aBits[i]).AsFloat32x8()
			var c int
			for ; c < wide; c += 32 {
				simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).MulAdd(vv, simd.LoadF32x8(outRow[c:])), outRow[c:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+8:]).MulAdd(vv, simd.LoadF32x8(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+16:]).MulAdd(vv, simd.LoadF32x8(outRow[c+16:])), outRow[c+16:])
				simd.StoreF32x8(simd.LoadF32x8(bRow[c+24:]).MulAdd(vv, simd.LoadF32x8(outRow[c+24:])), outRow[c+24:])
			}
			for ; c < vecs; c += 8 {
				simd.StoreF32x8(simd.LoadF32x8(bRow[c:]).MulAdd(vv, simd.LoadF32x8(outRow[c:])), outRow[c:])
			}
			if c < cols {
				simd.StoreF32x8Part(simd.LoadF32x8Part(bRow[c:]).MulAdd(vv, simd.LoadF32x8Part(outRow[c:])), outRow[c:])
			}
		}
	}
}
