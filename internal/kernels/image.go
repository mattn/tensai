package kernels

import "math"

// LayerNorm64 keeps the double-precision statistics and normalization used
// by the image transformer. dst may alias src.
func LayerNorm64(dst, src []float32, eps float64) {
	mean := Sum64(src) / float64(len(src))
	inv := 1 / math.Sqrt(SquaredDeviations64(src, mean)/float64(len(src))+eps)
	normalize64(dst, src, mean, inv)
}

// Softmax normalizes one finite, nonempty row in place.
func Softmax(row []float32) {
	if len(row) == 0 {
		return
	}
	ExpShift(row, row, maxSlice(row))
	ScaleSlice(row, float32(1/Sum64(row)))
}

func sum64Generic(x []float32) (s float64) {
	for _, v := range x {
		s += float64(v)
	}
	return
}
func squaredDeviations64Generic(x []float32, mean float64) (s float64) {
	for _, v := range x {
		d := float64(v) - mean
		s += d * d
	}
	return
}
func normalize64Generic(dst, src []float32, mean, inv float64) {
	for i, v := range src {
		dst[i] = float32((float64(v) - mean) * inv)
	}
}
func scaleWeightsGeneric(x, w []float32, scale, bias float32) {
	for i, v := range x {
		x[i] = v * scale * (w[i] + bias)
	}
}
func mulAddSliceGeneric(dst, x, y []float32) {
	for i := range dst {
		dst[i] += x[i] * y[i]
	}
}
func ropePairsGeneric(x, cos, sin []float32) {
	for j := range cos {
		a, b := x[2*j], x[2*j+1]
		x[2*j] = a*cos[j] - b*sin[j]
		x[2*j+1] = a*sin[j] + b*cos[j]
	}
}
func ropeSplitGeneric(x, cos, sin []float32) {
	half := len(x) / 2
	for j := range cos {
		a, b := x[j], x[half+j]
		x[j] = a*cos[j] - b*sin[j]
		x[half+j] = b*cos[j] + a*sin[j]
	}
}
func geluTanhGeneric(x []float32) {
	for i, v := range x {
		f := float64(v)
		x[i] = float32(0.5 * f * (1 + math.Tanh(0.7978845608028654*(f+0.044715*f*f*f))))
	}
}
func maxSliceGeneric(x []float32) float32 {
	m := float32(math.Inf(-1))
	for _, v := range x {
		if v > m {
			m = v
		}
	}
	return m
}
