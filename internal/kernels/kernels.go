// Package kernels holds the element-wise compute kernels shared by the
// tensai packages: scalar bodies here, with the exported entry points
// (ReluFwd, AdamStep, ...) defined per build in dispatch_generic.go and
// dispatch_simd.go, mirroring the dotRows split in the root package.
package kernels

import "math"

// float32 wrappers for the float64-only math package.
func ExpF(x float32) float32    { return float32(math.Exp(float64(x))) }
func LogF(x float32) float32    { return float32(math.Log(float64(x))) }
func TanhF(x float32) float32   { return float32(math.Tanh(float64(x))) }
func SqrtF(x float32) float32   { return float32(math.Sqrt(float64(x))) }
func PowF(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }

// GeluF is the exact GELU using erf.
func GeluF(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x/math.Sqrt2))))
}

// GeluGrad is the derivative of GeluF.
func GeluGrad(x float32) float32 {
	const invSqrt2Pi = 0.3989422804014327
	return 0.5*(1+float32(math.Erf(float64(x/math.Sqrt2)))) + x*float32(invSqrt2Pi)*ExpF(-0.5*x*x)
}

func reluFwdGeneric(dst, src []float32) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = 0
		}
	}
}

func reluBwdGeneric(dst, grad, src []float32) {
	for i, g := range grad {
		if src[i] > 0 {
			dst[i] = g
		} else {
			dst[i] = 0
		}
	}
}

func leakyFwdGeneric(dst, src []float32, alpha float32) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = alpha * v
		}
	}
}

func leakyBwdGeneric(dst, grad, src []float32, alpha float32) {
	for i, g := range grad {
		if src[i] > 0 {
			dst[i] = g
		} else {
			dst[i] = alpha * g
		}
	}
}

func sigmoidFwdGeneric(dst, src []float32) {
	for i, v := range src {
		dst[i] = 1 / (1 + ExpF(-v))
	}
}

// siluMulGeneric computes gate = silu(gate) * up, the SwiGLU gate.
func siluMulGeneric(gate, up []float32) {
	for i, g := range gate {
		gate[i] = g / (1 + ExpF(-g)) * up[i]
	}
}

// sigmoidBwdGeneric computes dst = grad * y * (1-y) from the forward output y.
func sigmoidBwdGeneric(dst, grad, y []float32) {
	for i, g := range grad {
		dst[i] = g * y[i] * (1 - y[i])
	}
}

func tanhFwdGeneric(dst, src []float32) {
	for i, v := range src {
		dst[i] = TanhF(v)
	}
}

// tanhBwdGeneric computes dst = grad * (1 - y^2) from the forward output y.
func tanhBwdGeneric(dst, grad, y []float32) {
	for i, g := range grad {
		dst[i] = g * (1 - y[i]*y[i])
	}
}

// expShiftGeneric computes dst = exp(src - shift), the softmax building block.
func expShiftGeneric(dst, src []float32, shift float32) {
	for i, v := range src {
		dst[i] = ExpF(v - shift)
	}
}

// addSliceGeneric computes dst += src.
func addSliceGeneric(dst, src []float32) {
	for i, v := range src {
		dst[i] += v
	}
}

// scaleSliceGeneric computes dst *= s.
func scaleSliceGeneric(dst []float32, s float32) {
	for i := range dst {
		dst[i] *= s
	}
}

// softmaxBwdAddGeneric accumulates the Jacobian-vector product
// dst += y * (grad-dot(grad,y)) for one softmax row.
func softmaxBwdAddGeneric(dst, grad, y []float32) {
	var dot float32
	for i, v := range y {
		dot += grad[i] * v
	}
	for i, v := range y {
		dst[i] += v * (grad[i] - dot)
	}
}

// adamStepGeneric applies one Adam/AdamW update over a parameter slice.
// rc1/rc2 are the reciprocal bias corrections 1/(1-beta^t).
func adamStepGeneric(w, g, m, v []float32, beta1, beta2, rc1, rc2, lr, eps, wd float32) {
	ib1 := 1 - beta1
	ib2 := 1 - beta2
	for i, wi := range w {
		gi := g[i]
		mi := beta1*m[i] + ib1*gi
		vi := beta2*v[i] + ib2*gi*gi
		m[i] = mi
		v[i] = vi
		w[i] = wi - lr*(mi*rc1/(SqrtF(vi*rc2)+eps)+wd*wi)
	}
}

// sgdStepGeneric applies one momentum-SGD update over a parameter slice:
// vel = momentum*vel - lr*g, w += vel.
func sgdStepGeneric(w, g, vel []float32, momentum, lr float32) {
	for i, wi := range w {
		vi := momentum*vel[i] - lr*g[i]
		vel[i] = vi
		w[i] = wi + vi
	}
}

// geluFwdGeneric computes dst = 0.5*src*(1+erf(src/sqrt(2))).
func geluFwdGeneric(dst, src []float32) {
	for i, v := range src {
		dst[i] = GeluF(v)
	}
}

// geluBwdGeneric computes dst = grad * gelu'(src).
func geluBwdGeneric(dst, grad, src []float32) {
	for i, g := range grad {
		dst[i] = g * GeluGrad(src[i])
	}
}

// lnFwdRowGeneric normalizes one row: writes the normalized values into
// xhat and gamma*xhat+beta into out, returning 1/sqrt(variance+eps).
func lnFwdRowGeneric(out, xhat, src, gamma, beta []float32, eps float32) float32 {
	n := float32(len(src))
	var mean float32
	for _, v := range src {
		mean += v
	}
	mean /= n
	var variance float32
	for _, v := range src {
		d := v - mean
		variance += d * d
	}
	variance /= n
	invStd := 1 / SqrtF(variance+eps)
	for c, v := range src {
		h := (v - mean) * invStd
		xhat[c] = h
		out[c] = h*gamma[c] + beta[c]
	}
	return invStd
}

// lnBwdRowGeneric runs one row of the LayerNorm backward pass: accumulates
// the parameter gradients and writes the input gradient into out.
func lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta []float32, invStd float32) {
	n := float32(len(g))
	var sumDXhat, sumDXhatXhat float32
	for c, gv := range g {
		gradGamma[c] += gv * xhat[c]
		gradBeta[c] += gv
		dxhat := gv * gamma[c]
		sumDXhat += dxhat
		sumDXhatXhat += dxhat * xhat[c]
	}
	k := invStd / n
	for c, gv := range g {
		dxhat := gv * gamma[c]
		out[c] = k * (n*dxhat - sumDXhat - xhat[c]*sumDXhatXhat)
	}
}

// dotVecGeneric and axpyGeneric are the scalar bodies of the public
// DotVec and Axpy vector helpers.
func dotVecGeneric(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func axpyGeneric(a float32, x, y []float32) {
	for i := range x {
		y[i] += a * x[i]
	}
}

// dotVecsGeneric and axpysGeneric are the scalar bodies of the grouped
// DotVecs and Axpys attention helpers.
func dotVecsGeneric(qs, k []float32, out []float32) {
	d := len(k)
	for i := range out {
		out[i] = dotVecGeneric(qs[i*d:(i+1)*d], k)
	}
}

func axpysGeneric(ws []float32, v, outs []float32) {
	d := len(v)
	for i := range ws {
		axpyGeneric(ws[i], v, outs[i*d:(i+1)*d])
	}
}
