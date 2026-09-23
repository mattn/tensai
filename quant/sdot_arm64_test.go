//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/mattn/tensai"
)

// The SDOT kernel and the widening one accumulate the same products in a
// different order, which for exact integers is the same sum: the two must
// agree to the last bit, and a machine without FEAT_DotProd must reach
// the same answer through the fallback.
func TestQMatVecDotProdMatchesWidening(t *testing.T) {
	if !hasDotProd {
		t.Skip("no FEAT_DotProd on this machine; only the widening path runs here")
	}
	rng := rand.New(rand.NewPCG(11, 0))
	for _, shape := range [][2]int{{896, 1152}, {896, 4864}, {4864, 896}, {128, 40}} {
		q := Quantize(tensai.RandomMatrix(shape[0], shape[1], rng))
		x := make([]tensai.Float, shape[0])
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
		}
		dot := make([]tensai.Float, shape[1])
		if err := q.MatVec(x, dot); err != nil {
			t.Fatal(err)
		}
		hasDotProd = false
		wide := make([]tensai.Float, shape[1])
		err := q.MatVec(x, wide)
		hasDotProd = true
		if err != nil {
			t.Fatal(err)
		}
		for i := range wide {
			if dot[i] != wide[i] {
				t.Fatalf("%dx%d column %d: sdot %v != widening %v", shape[0], shape[1], i, dot[i], wide[i])
			}
		}
	}
}

// BenchmarkQMatVecDotProd puts the two paths side by side on the weight
// shapes a decode step walks. Emulation cannot answer what SDOT is
// worth, so the number this prints is only meaningful from the macOS
// runner or another arm64 machine, where the two rows are measured the
// same way and differ in the fold alone.
func BenchmarkQMatVecDotProd(b *testing.B) {
	if !hasDotProd {
		b.Skip("no FEAT_DotProd on this machine; only the widening path runs here")
	}
	rng := rand.New(rand.NewPCG(11, 0))
	for _, shape := range [][2]int{{896, 1152}, {896, 4864}, {4864, 896}} {
		q := Quantize(tensai.RandomMatrix(shape[0], shape[1], rng))
		x := make([]tensai.Float, shape[0])
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
		}
		out := make([]tensai.Float, shape[1])
		run := func(b *testing.B) {
			b.SetBytes(int64(shape[0]) * int64(shape[1]))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := q.MatVec(x, out); err != nil {
					b.Fatal(err)
				}
			}
		}
		name := fmt.Sprintf("%dx%d", shape[0], shape[1])
		b.Run(name+"/sdot", run)
		b.Run(name+"/widening", func(b *testing.B) {
			hasDotProd = false
			defer func() { hasDotProd = true }()
			run(b)
		})
	}
}
