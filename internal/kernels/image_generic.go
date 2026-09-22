//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package kernels

// Sum64 adds float32 values with float64 accumulation.
func Sum64(x []float32) float64 { return sum64Generic(x) }

// SquaredDeviations64 sums (x-mean)^2 in float64; mean=0 gives squared norm.
func SquaredDeviations64(x []float32, mean float64) float64 {
	return squaredDeviations64Generic(x, mean)
}
func normalize64(dst, src []float32, mean, inv float64) { normalize64Generic(dst, src, mean, inv) }

// ScaleWeights computes x[i] = x[i] * scale * (w[i] + bias), in place.
func ScaleWeights(x, w []float32, scale, bias float32) { scaleWeightsGeneric(x, w, scale, bias) }

// MulAddSlice computes dst += x*y; dst may alias either input.
func MulAddSlice(dst, x, y []float32) { mulAddSliceGeneric(dst, x, y) }

// RopePairs rotates adjacent pairs. cos and sin each have len(x)/2 elements.
func RopePairs(x, cos, sin []float32) { ropePairsGeneric(x, cos, sin) }

// RopeSplit rotates the two halves of x against each other.
func RopeSplit(x, cos, sin []float32) { ropeSplitGeneric(x, cos, sin) }

// GeluTanh applies the tanh approximation of GELU in place.
func GeluTanh(x []float32)         { geluTanhGeneric(x) }
func maxSlice(x []float32) float32 { return maxSliceGeneric(x) }
