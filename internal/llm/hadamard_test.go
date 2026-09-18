package llm

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"

	"github.com/mattn/tensai/internal/kernels"
)

// fwht is the Sylvester matrix: entry (r, c) is (-1)^popcount(r&c)
// over sqrt(n), which is how the reference builds its rotation.
func TestFWHTIsSylvester(t *testing.T) {
	const n = 16
	for c := 0; c < n; c++ {
		v := make([]float32, n)
		v[c] = 1
		fwht(v)
		for r := 0; r < n; r++ {
			want := 1 / math.Sqrt(n)
			if bits.OnesCount(uint(r&c))%2 == 1 {
				want = -want
			}
			if math.Abs(float64(v[r])-want) > 1e-6 {
				t.Fatalf("H[%d][%d] = %v, want %v", r, c, v[r], want)
			}
		}
	}
}

// Applying the transform and then its inverse is the identity, blocks,
// signs and permutation included, and the transform preserves length.
func TestHadamardRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const block, width = 8, 48
	h := &hadamard{block: block, signs: make([]float32, width)}
	for i := range h.signs {
		h.signs[i] = float32(1 - 2*rng.Intn(2))
	}
	x := make([]float32, width)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := h.apply(x)
	var nx, ny float64
	for i := range x {
		nx += float64(x[i] * x[i])
		ny += float64(y[i] * y[i])
	}
	if math.Abs(nx-ny) > 1e-4*nx {
		t.Fatalf("norm %v became %v", nx, ny)
	}
	h.invert(y)
	for i := range x {
		if math.Abs(float64(y[i]-x[i])) > 1e-5 {
			t.Fatalf("round trip changed x[%d] from %v to %v", i, x[i], y[i])
		}
	}
	// The permutation moves tiled value heads to grouped order: with
	// two key heads, three replicas and a head width of two, tiled
	// position (replica j, key head g) lands at grouped (g, j).
	perm := tiledToGrouped(2, 2, 3)
	src := make([]float32, 12)
	for j := 0; j < 3; j++ {
		for g := 0; g < 2; g++ {
			for d := 0; d < 2; d++ {
				src[(j*2+g)*2+d] = float32(100*g + 10*j + d)
			}
		}
	}
	dst := make([]float32, 12)
	for i, p := range perm {
		dst[i] = src[p]
	}
	for g := 0; g < 2; g++ {
		for j := 0; j < 3; j++ {
			for d := 0; d < 2; d++ {
				if got := dst[(g*3+j)*2+d]; got != float32(100*g+10*j+d) {
					t.Fatalf("grouped (%d,%d,%d) = %v", g, j, d, got)
				}
			}
		}
	}
}

// A rotated weight applied to an activation equals the plain weight
// applied to the rotated activation; the loader folds the rotation on
// one side and the runtime supplies the other.
func TestRotatedWeightReadsRotatedInput(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const block, in, out = 4, 8, 3
	h := &hadamard{block: block, signs: []float32{1, -1, 1, 1, -1, -1, 1, -1}}
	w := make([][]float32, out) // [out][in], the HF orientation
	for o := range w {
		w[o] = make([]float32, in)
		for i := range w[o] {
			w[o][i] = float32(rng.NormFloat64())
		}
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	// Fold the rotation into the weights: w' = w H S, i.e. each row of
	// w transformed the way an input is (H and S are symmetric).
	wr := make([][]float32, out)
	for o := range w {
		wr[o] = h.apply(w[o])
	}
	xr := h.apply(x)
	for o := 0; o < out; o++ {
		var plain, rot float64
		for i := 0; i < in; i++ {
			plain += float64(w[o][i] * x[i])
			rot += float64(wr[o][i] * xr[i])
		}
		if math.Abs(plain-rot) > 1e-4 {
			t.Fatalf("output %d: plain %v, rotated %v", o, plain, rot)
		}
	}
}

func BenchmarkHadamardApply(b *testing.B) {
	const width = 17408
	h := &hadamard{block: 1024, signs: make([]float32, width)}
	for i := range h.signs {
		h.signs[i] = float32(1 - 2*(i%3&1))
	}
	x := make([]float32, width)
	for i := range x {
		x[i] = float32(i%13) - 6
	}
	b.SetBytes(width * 4)
	for i := 0; i < b.N; i++ {
		h.release(h.apply(x))
	}
}

func BenchmarkHadamardParts(b *testing.B) {
	const width = 17408
	h := &hadamard{block: 1024, signs: make([]float32, width)}
	x := make([]float32, width)
	y := make([]float32, width)
	b.Run("mulsigns", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			kernels.MulSlices(y, x, h.signs)
		}
	})
	b.Run("blocks", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for off := 0; off+1024 <= width; off += 1024 {
				kernels.Hadamard(y[off:off+1024], 0.03125)
			}
		}
	})
	b.Run("scale", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			kernels.ScaleSlice(y, 0.5)
		}
	})
	b.Run("alloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			y = make([]float32, width)
		}
	})
}
