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

// adamStepGeneric applies one Adam/AdamW update over a parameter slice.
// rc1/rc2 are the reciprocal bias corrections 1/(1-beta^t).
func adamStepGeneric(w, g, m, v []Float, beta1, beta2, rc1, rc2, lr, eps, wd Float) {
	for i := range w {
		gi := g[i]
		m[i] = beta1*m[i] + (1-beta1)*gi
		v[i] = beta2*v[i] + (1-beta2)*gi*gi
		mHat := m[i] * rc1
		vHat := v[i] * rc2
		w[i] -= lr * (mHat/(sqrtF(vHat)+eps) + wd*w[i])
	}
}
