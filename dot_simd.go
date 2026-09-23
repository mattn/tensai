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

	// Four or more rows share b: tile them so each loaded row of b feeds
	// four rows of out held in registers, and b is streamed once per
	// block of rows instead of once per row. Whatever the tiles leave --
	// a column tail short of 16, the last rows short of 4 -- goes through
	// the row-at-a-time kernel.
	cols := b.Cols
	if hi-lo >= 4 && cols >= 16 {
		r4 := lo + (hi-lo)&^3
		n16 := cols &^ 15
		dotRowsTiled(out, a, b, lo, r4, n16)
		if n16 < cols {
			dotRowsAxpy(out, a, b, lo, r4, n16)
		}
		lo = r4
	}
	dotRowsAxpy(out, a, b, lo, hi, 0)
}

// dotRowsAxpy computes rows lo..hi, columns c0..b.Cols of out = a * b one
// row at a time: every nonzero element of a's row scales the matching row
// of b into the output row.
func dotRowsAxpy(out, a, b *Matrix, lo, hi, c0 int) {
	cols := b.Cols
	width := cols - c0
	wide := width &^ 31 // widest multiple of 32
	vecs := width &^ 7  // widest multiple of 8
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		aBits := unsafe.Slice((*uint32)(unsafe.Pointer(&aRow[0])), len(aRow))
		outRow := out.Data[r*cols+c0 : (r+1)*cols]
		initialized := false
		for k := range aBits {
			if aBits[k]<<1 == 0 { // +0.0 or -0.0
				continue
			}
			bRow := b.Data[k*cols+c0 : (k+1)*cols]
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
				if c < width {
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
			if c < width {
				simd.StoreF32x8Part(simd.LoadF32x8Part(bRow[c:]).MulAdd(vv, simd.LoadF32x8Part(outRow[c:])), outRow[c:])
			}
		}
		if !initialized {
			clear(outRow)
		}
	}
}

// dotTileK is how many rows of b one pass of dotRowsTiled walks: a 16-wide
// strip of that many rows is 16KB, small enough to stay in L1/L2 while
// every block of four output rows reuses it.
const dotTileK = 256

// dotRowsTiled computes rows lo..hi (a multiple of 4 apart), columns
// 0..n16 (a multiple of 16) of out = a * b in 4x16 register tiles. Each
// tile accumulates in k order with FMAs starting from zero, so a finite
// result comes out bit for bit as dotRowsAxpy leaves it: a zero element of
// a adds an exact zero where the row kernel skips it. Only the sign of a
// zero result can differ, and an Inf or NaN in b meeting a zero in a,
// which the row kernel never multiplies.
func dotRowsTiled(out, a, b *Matrix, lo, hi, n16 int) {
	for k0 := 0; k0 < a.Cols; k0 += dotTileK {
		k1 := min(k0+dotTileK, a.Cols)
		for c := 0; c < n16; c += 16 {
			for r := lo; r < hi; r += 4 {
				dotTile4x16(out, a, b, r, c, k0, k1)
			}
		}
	}
}

// dotTile4x16 adds a[r:r+4, k0:k1] * b[k0:k1, c:c+16] into the 4x16 tile of
// out at (r, c), or writes it there when k0 is 0. Eight accumulators, two
// loads of b and one broadcast per row keep it inside the 16 vector
// registers.
func dotTile4x16(out, a, b *Matrix, r, c, k0, k1 int) {
	n, kk := b.Cols, a.Cols
	o0 := out.Data[r*n+c : r*n+c+16]
	o1 := out.Data[(r+1)*n+c : (r+1)*n+c+16]
	o2 := out.Data[(r+2)*n+c : (r+2)*n+c+16]
	o3 := out.Data[(r+3)*n+c : (r+3)*n+c+16]
	var c00, c01, c10, c11, c20, c21, c30, c31 archsimd.Float32x8
	if k0 > 0 {
		c00, c01 = simd.LoadF32x8(o0), simd.LoadF32x8(o0[8:])
		c10, c11 = simd.LoadF32x8(o1), simd.LoadF32x8(o1[8:])
		c20, c21 = simd.LoadF32x8(o2), simd.LoadF32x8(o2[8:])
		c30, c31 = simd.LoadF32x8(o3), simd.LoadF32x8(o3[8:])
	}
	a0 := a.Data[r*kk+k0 : r*kk+k1]
	a1 := a.Data[(r+1)*kk+k0 : (r+1)*kk+k1]
	a2 := a.Data[(r+2)*kk+k0 : (r+2)*kk+k1]
	a3 := a.Data[(r+3)*kk+k0 : (r+3)*kk+k1]
	a1, a2, a3 = a1[:len(a0)], a2[:len(a0)], a3[:len(a0)]
	bi := k0*n + c
	for k := range a0 {
		bRow := b.Data[bi : bi+16]
		bi += n
		b0, b1 := simd.LoadF32x8(bRow), simd.LoadF32x8(bRow[8:])
		v := archsimd.BroadcastFloat32x8(a0[k])
		c00, c01 = b0.MulAdd(v, c00), b1.MulAdd(v, c01)
		v = archsimd.BroadcastFloat32x8(a1[k])
		c10, c11 = b0.MulAdd(v, c10), b1.MulAdd(v, c11)
		v = archsimd.BroadcastFloat32x8(a2[k])
		c20, c21 = b0.MulAdd(v, c20), b1.MulAdd(v, c21)
		v = archsimd.BroadcastFloat32x8(a3[k])
		c30, c31 = b0.MulAdd(v, c30), b1.MulAdd(v, c31)
	}
	simd.StoreF32x8(c00, o0)
	simd.StoreF32x8(c01, o0[8:])
	simd.StoreF32x8(c10, o1)
	simd.StoreF32x8(c11, o1[8:])
	simd.StoreF32x8(c20, o2)
	simd.StoreF32x8(c21, o2[8:])
	simd.StoreF32x8(c30, o3)
	simd.StoreF32x8(c31, o3[8:])
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
