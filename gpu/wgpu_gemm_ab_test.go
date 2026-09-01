//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu

import (
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkGemmWide compares the vec4-load kernel against the scalar-load
// one in a single process, alternating between them: this machine drifts
// enough between runs that separate ones cannot tell a 10% difference.
func BenchmarkGemmWide(b *testing.B) {
	g, err := Open()
	if err != nil {
		b.Skipf("wgpu unavailable: %v", err)
	}
	defer g.Close()
	rng := rand.New(rand.NewSource(29))
	for _, size := range []int{512, 1024, 2048} {
		x, err := g.Upload(randTensor(rng, size, size))
		if err != nil {
			b.Fatal(err)
		}
		w, err := g.Upload(randTensor(rng, size, size))
		if err != nil {
			b.Fatal(err)
		}
		run := func(off bool) func(*testing.B) {
			return func(b *testing.B) {
				noWideGemm = off
				defer func() { noWideGemm = false }()
				for i := 0; i < b.N; i++ {
					out, err := x.MatMul(w)
					if err != nil {
						b.Fatal(err)
					}
					out.Free()
				}
				if out, err := x.MatMul(w); err == nil {
					out.Download()
					out.Free()
				}
			}
		}
		b.Run(fmt.Sprintf("scalar/%d", size), run(true))
		b.Run(fmt.Sprintf("vec4/%d", size), run(false))
		b.Run(fmt.Sprintf("scalar2/%d", size), run(true))
		b.Run(fmt.Sprintf("vec4b/%d", size), run(false))
		x.Free()
		w.Free()
	}
}
