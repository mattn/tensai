//go:build goexperiment.simd && arm64 && go1.27

package tensai

import (
	"runtime"
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// The dense float matmuls on arm64: the amd64 kernels of dot_simd.go over
// 4-lane NEON vectors, unrolled four deep so a row step covers sixteen
// columns, with part loads for the column tail. The multiply-add is
// fused, as the arm64 compiler fuses the portable bodies' `out += a*b`.

// dotRows computes rows lo..hi of out = a * b.
func dotRows(out, a, b *Matrix, lo, hi int) {
	cols := b.Cols
	wide := cols &^ 15 // widest multiple of 16
	vecs := cols &^ 3  // widest multiple of 4
	for r := lo; r < hi; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		outRow := out.Data[r*cols : (r+1)*cols]
		initialized := false
		for k, av := range aRow {
			if av == 0 {
				continue
			}
			bRow := b.Data[k*cols : (k+1)*cols]
			vv := archsimd.BroadcastFloat32x4(av)
			var c int
			if !initialized {
				for ; c < wide; c += 16 {
					simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).Mul(vv), outRow[c:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).Mul(vv), outRow[c+4:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).Mul(vv), outRow[c+8:])
					simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).Mul(vv), outRow[c+12:])
				}
				for ; c < vecs; c += 4 {
					simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).Mul(vv), outRow[c:])
				}
				if c < cols {
					simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).Mul(vv), outRow[c:])
				}
				initialized = true
				continue
			}
			for ; c < wide; c += 16 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).MulAdd(vv, simd.LoadF32x4(outRow[c+4:])), outRow[c+4:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).MulAdd(vv, simd.LoadF32x4(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).MulAdd(vv, simd.LoadF32x4(outRow[c+12:])), outRow[c+12:])
			}
			for ; c < vecs; c += 4 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
			}
			if c < cols {
				simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).MulAdd(vv, simd.LoadF32x4Part(outRow[c:])), outRow[c:])
			}
		}
		if !initialized {
			clear(outRow)
		}
	}
}

func dotWorkerCount(rows, inner, cols int) int {
	workers := 1
	if rows*inner*cols >= 1<<20 {
		workers = runtime.NumCPU()
		if workers > 8 {
			workers = 8
		}
		if workers > rows {
			workers = rows
		}
	}
	return workers
}

// dotTATall computes out = a^T * b when b has at most eight columns: the
// output rows lo..hi are accumulated four at a time in vector registers,
// two vectors per row, so the inputs are streamed instead of the output
// being read back and written for every element.
func dotTATall(out, a, b *Matrix, lo, hi int) {
	k, n, rows := a.Cols, b.Cols, a.Rows
	for j0 := 0; j0 < n; j0 += 8 {
		width := min(8, n-j0)
		dotTATallCols(out, a, b, lo, hi, k, n, rows, j0, width)
	}
}

// dotTATallCols is dotTATall over one eight-wide slice of b's columns,
// two vectors per output row; part loads and stores cover a slice that
// is not a whole eight, and the second vector is empty when it is four
// or fewer.
func dotTATallCols(out, a, b *Matrix, lo, hi, k, n, rows, j0, width int) {
	w0 := min(width, 4)
	for i0 := lo; i0 < hi; i0 += 4 {
		var a0, a1, a2, a3, c0, c1, c2, c3 archsimd.Float32x4
		switch hi - i0 {
		case 1:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
			}
		case 2:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
			}
		case 3:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
				v = archsimd.BroadcastFloat32x4(aRow[2])
				a2, c2 = b0.MulAdd(v, a2), b1.MulAdd(v, c2)
			}
		default:
			for r := 0; r < rows; r++ {
				row := b.Data[r*n+j0 : r*n+j0+width]
				b0, b1 := simd.LoadF32x4Part(row[:w0]), simd.LoadF32x4Part(row[w0:])
				aRow := a.Data[r*k+i0:]
				v := archsimd.BroadcastFloat32x4(aRow[0])
				a0, c0 = b0.MulAdd(v, a0), b1.MulAdd(v, c0)
				v = archsimd.BroadcastFloat32x4(aRow[1])
				a1, c1 = b0.MulAdd(v, a1), b1.MulAdd(v, c1)
				v = archsimd.BroadcastFloat32x4(aRow[2])
				a2, c2 = b0.MulAdd(v, a2), b1.MulAdd(v, c2)
				v = archsimd.BroadcastFloat32x4(aRow[3])
				a3, c3 = b0.MulAdd(v, a3), b1.MulAdd(v, c3)
			}
		}
		lo4 := [4]archsimd.Float32x4{a0, a1, a2, a3}
		hi4 := [4]archsimd.Float32x4{c0, c1, c2, c3}
		for i := i0; i < hi && i < i0+4; i++ {
			row := out.Data[i*n+j0 : i*n+j0+width]
			simd.StoreF32x4Part(lo4[i-i0], row[:w0])
			simd.StoreF32x4Part(hi4[i-i0], row[w0:])
		}
	}
}

// dotTARows computes out rows lo..hi of out = a^T * b: a's element is
// broadcast against b's row and accumulated into out's row.
func dotTARows(out, a, b *Matrix, lo, hi int) {
	cols := b.Cols
	wide := cols &^ 15
	vecs := cols &^ 3
	for r := 0; r < a.Rows; r++ {
		aRow := a.Data[r*a.Cols : (r+1)*a.Cols]
		bRow := b.Data[r*cols : (r+1)*cols]
		for i := lo; i < hi; i++ {
			av := aRow[i]
			if av == 0 {
				continue
			}
			outRow := out.Data[i*cols : (i+1)*cols]
			vv := archsimd.BroadcastFloat32x4(av)
			var c int
			for ; c < wide; c += 16 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+4:]).MulAdd(vv, simd.LoadF32x4(outRow[c+4:])), outRow[c+4:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+8:]).MulAdd(vv, simd.LoadF32x4(outRow[c+8:])), outRow[c+8:])
				simd.StoreF32x4(simd.LoadF32x4(bRow[c+12:]).MulAdd(vv, simd.LoadF32x4(outRow[c+12:])), outRow[c+12:])
			}
			for ; c < vecs; c += 4 {
				simd.StoreF32x4(simd.LoadF32x4(bRow[c:]).MulAdd(vv, simd.LoadF32x4(outRow[c:])), outRow[c:])
			}
			if c < cols {
				simd.StoreF32x4Part(simd.LoadF32x4Part(bRow[c:]).MulAdd(vv, simd.LoadF32x4Part(outRow[c:])), outRow[c:])
			}
		}
	}
}
