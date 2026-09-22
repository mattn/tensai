package qwenimage

import (
	"math"
	"testing"

	"github.com/mattn/tensai"
)

// Keep the original rotation as an independent reference: both the table
// construction and the split-half convention are observable here.
func textRopeScalar(x *tensai.Matrix, heads int) {
	const half = teHeadDim / 2
	for r := 0; r < x.Rows; r++ {
		for h := 0; h < heads; h++ {
			head := x.Data[r*x.Cols+h*teHeadDim:][:teHeadDim]
			for j := 0; j < half; j++ {
				angle := float64(r) * math.Pow(teRopeTheta, -2*float64(j)/teHeadDim)
				c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
				a, b := head[j], head[half+j]
				head[j] = a*c - b*s
				head[half+j] = b*c + a*s
			}
		}
	}
}
func TestTextRopeTable(t *testing.T) {
	for _, heads := range []int{teHeads, teKVHeads} {
		want := tensai.NewMatrix(25, heads*teHeadDim)
		for i := range want.Data {
			want.Data[i] = float32(math.Sin(float64(i) * 0.17))
		}
		got := clone(want)
		textRopeScalar(want, heads)
		cos, sin := teRopeTable(got.Rows)
		teRopeWithTable(got, heads, cos, sin)
		for i, v := range want.Data {
			if d := math.Abs(float64(got.Data[i] - v)); math.IsNaN(d) || d > 2e-6 {
				t.Fatalf("%d heads, element %d: got %g want %g", heads, i, got.Data[i], v)
			}
		}
	}
}
func BenchmarkTextRope(b *testing.B) {
	x := tensai.NewMatrix(25, teDim)
	for i := range x.Data {
		x.Data[i] = float32(math.Sin(float64(i) * 0.17))
	}
	cos, sin := teRopeTable(x.Rows)
	b.Run("before", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			textRopeScalar(x, teHeads)
		}
	})
	b.Run("cached_SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			teRopeWithTable(x, teHeads, cos, sin)
		}
	})
}
