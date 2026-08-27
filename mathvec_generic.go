//go:build !goexperiment.simd || !amd64

package tensai

// Portable dispatchers for the element-wise kernels; build with
// GOEXPERIMENT=simd on amd64 for the AVX2 versions in mathvec_simd.go.

func reluFwd(dst, src []Float)               { reluFwdGeneric(dst, src) }
func reluBwd(dst, grad, src []Float)         { reluBwdGeneric(dst, grad, src) }
func leakyFwd(dst, src []Float, alpha Float) { leakyFwdGeneric(dst, src, alpha) }
func leakyBwd(dst, grad, src []Float, alpha Float) {
	leakyBwdGeneric(dst, grad, src, alpha)
}
func sigmoidFwd(dst, src []Float)            { sigmoidFwdGeneric(dst, src) }
func siluMul(gate, up []Float)               { siluMulGeneric(gate, up) }
func sigmoidBwd(dst, grad, y []Float)        { sigmoidBwdGeneric(dst, grad, y) }
func tanhFwd(dst, src []Float)               { tanhFwdGeneric(dst, src) }
func tanhBwd(dst, grad, y []Float)           { tanhBwdGeneric(dst, grad, y) }
func expShift(dst, src []Float, shift Float) { expShiftGeneric(dst, src, shift) }
func addSlice(dst, src []Float)              { addSliceGeneric(dst, src) }
func scaleSlice(dst []Float, s Float)        { scaleSliceGeneric(dst, s) }
func softmaxBwdAdd(dst, grad, y []Float)     { softmaxBwdAddGeneric(dst, grad, y) }
func adamStepSlice(w, g, m, v []Float, beta1, beta2, rc1, rc2, lr, eps, wd Float) {
	adamStepGeneric(w, g, m, v, beta1, beta2, rc1, rc2, lr, eps, wd)
}
func sgdStepSlice(w, g, vel []Float, momentum, lr Float) {
	sgdStepGeneric(w, g, vel, momentum, lr)
}

func geluFwd(dst, src []Float)       { geluFwdGeneric(dst, src) }
func geluBwd(dst, grad, src []Float) { geluBwdGeneric(dst, grad, src) }
func lnFwdRow(out, xhat, src, gamma, beta []Float, eps Float) Float {
	return lnFwdRowGeneric(out, xhat, src, gamma, beta, eps)
}
func lnBwdRow(out, g, xhat, gamma, gradGamma, gradBeta []Float, invStd Float) {
	lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta, invStd)
}

func dotVec(a, b []Float) Float  { return dotVecGeneric(a, b) }
func axpy(a Float, x, y []Float) { axpyGeneric(a, x, y) }

func dotVecs(qs, k []Float, out []Float) { dotVecsGeneric(qs, k, out) }

func axpys(ws []Float, v, outs []Float) { axpysGeneric(ws, v, outs) }
