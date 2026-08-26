package tensai

// Scalar bodies for the element-wise kernels. The exported-in-spirit entry
// points (reluFwd, adamStepSlice, ...) are defined per build in
// mathvec_generic.go and mathvec_simd.go, mirroring the dotRows split.

func reluFwdGeneric(dst, src []Float) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = 0
		}
	}
}

func reluBwdGeneric(dst, grad, src []Float) {
	for i, g := range grad {
		if src[i] > 0 {
			dst[i] = g
		} else {
			dst[i] = 0
		}
	}
}

func leakyFwdGeneric(dst, src []Float, alpha Float) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = alpha * v
		}
	}
}

func leakyBwdGeneric(dst, grad, src []Float, alpha Float) {
	for i, g := range grad {
		if src[i] > 0 {
			dst[i] = g
		} else {
			dst[i] = alpha * g
		}
	}
}

func sigmoidFwdGeneric(dst, src []Float) {
	for i, v := range src {
		dst[i] = 1 / (1 + expF(-v))
	}
}

// siluMulGeneric computes gate = silu(gate) * up, the SwiGLU gate.
func siluMulGeneric(gate, up []Float) {
	for i, g := range gate {
		gate[i] = g / (1 + expF(-g)) * up[i]
	}
}

// sigmoidBwdGeneric computes dst = grad * y * (1-y) from the forward output y.
func sigmoidBwdGeneric(dst, grad, y []Float) {
	for i, g := range grad {
		dst[i] = g * y[i] * (1 - y[i])
	}
}

func tanhFwdGeneric(dst, src []Float) {
	for i, v := range src {
		dst[i] = tanhF(v)
	}
}

// tanhBwdGeneric computes dst = grad * (1 - y^2) from the forward output y.
func tanhBwdGeneric(dst, grad, y []Float) {
	for i, g := range grad {
		dst[i] = g * (1 - y[i]*y[i])
	}
}

// expShiftGeneric computes dst = exp(src - shift), the softmax building block.
func expShiftGeneric(dst, src []Float, shift Float) {
	for i, v := range src {
		dst[i] = expF(v - shift)
	}
}

// addSliceGeneric computes dst += src.
func addSliceGeneric(dst, src []Float) {
	for i, v := range src {
		dst[i] += v
	}
}

// scaleSliceGeneric computes dst *= s.
func scaleSliceGeneric(dst []Float, s Float) {
	for i := range dst {
		dst[i] *= s
	}
}

// softmaxBwdAddGeneric accumulates the Jacobian-vector product
// dst += y * (grad-dot(grad,y)) for one softmax row.
func softmaxBwdAddGeneric(dst, grad, y []Float) {
	var dot Float
	for i, v := range y {
		dot += grad[i] * v
	}
	for i, v := range y {
		dst[i] += v * (grad[i] - dot)
	}
}

// adamStepGeneric applies one Adam/AdamW update over a parameter slice.
// rc1/rc2 are the reciprocal bias corrections 1/(1-beta^t).
func adamStepGeneric(w, g, m, v []Float, beta1, beta2, rc1, rc2, lr, eps, wd Float) {
	ib1 := 1 - beta1
	ib2 := 1 - beta2
	for i, wi := range w {
		gi := g[i]
		mi := beta1*m[i] + ib1*gi
		vi := beta2*v[i] + ib2*gi*gi
		m[i] = mi
		v[i] = vi
		w[i] = wi - lr*(mi*rc1/(sqrtF(vi*rc2)+eps)+wd*wi)
	}
}

// geluFwdGeneric computes dst = 0.5*src*(1+erf(src/sqrt(2))).
func geluFwdGeneric(dst, src []Float) {
	for i, v := range src {
		dst[i] = geluF(v)
	}
}

// geluBwdGeneric computes dst = grad * gelu'(src).
func geluBwdGeneric(dst, grad, src []Float) {
	for i, g := range grad {
		dst[i] = g * geluGrad(src[i])
	}
}

// lnFwdRowGeneric normalizes one row: writes the normalized values into
// xhat and gamma*xhat+beta into out, returning 1/sqrt(variance+eps).
func lnFwdRowGeneric(out, xhat, src, gamma, beta []Float, eps Float) Float {
	n := Float(len(src))
	var mean Float
	for _, v := range src {
		mean += v
	}
	mean /= n
	var variance Float
	for _, v := range src {
		d := v - mean
		variance += d * d
	}
	variance /= n
	invStd := 1 / sqrtF(variance+eps)
	for c, v := range src {
		h := (v - mean) * invStd
		xhat[c] = h
		out[c] = h*gamma[c] + beta[c]
	}
	return invStd
}

// lnBwdRowGeneric runs one row of the LayerNorm backward pass: accumulates
// the parameter gradients and writes the input gradient into out.
func lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta []Float, invStd Float) {
	n := Float(len(g))
	var sumDXhat, sumDXhatXhat Float
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
func dotVecGeneric(a, b []Float) Float {
	var s Float
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func axpyGeneric(a Float, x, y []Float) {
	for i := range x {
		y[i] += a * x[i]
	}
}

// dotVecsGeneric and axpysGeneric are the scalar bodies of the grouped
// DotVecs and Axpys attention helpers.
func dotVecsGeneric(qs, k []Float, out []Float) {
	d := len(k)
	for i := range out {
		out[i] = dotVecGeneric(qs[i*d:(i+1)*d], k)
	}
}

func axpysGeneric(ws []Float, v, outs []Float) {
	d := len(v)
	for i := range ws {
		axpyGeneric(ws[i], v, outs[i*d:(i+1)*d])
	}
}
