//go:build !goexperiment.simd || (!amd64 && (!arm64 || !go1.27))

package kernels

// Portable dispatchers for the element-wise kernels; build with
// GOEXPERIMENT=simd for the AVX2 versions in dispatch_simd.go on amd64
// or the NEON versions in dispatch_neon.go on arm64.

func ReluFwd(dst, src []float32)                 { reluFwdGeneric(dst, src) }
func ReluBwd(dst, grad, src []float32)           { reluBwdGeneric(dst, grad, src) }
func LeakyFwd(dst, src []float32, alpha float32) { leakyFwdGeneric(dst, src, alpha) }
func LeakyBwd(dst, grad, src []float32, alpha float32) {
	leakyBwdGeneric(dst, grad, src, alpha)
}
func SigmoidFwd(dst, src []float32)              { sigmoidFwdGeneric(dst, src) }
func SiluMul(gate, up []float32)                 { siluMulGeneric(gate, up) }
func GeluMul(gate, up []float32)                 { geluMulGeneric(gate, up) }
func Silu(v []float32)                           { siluGeneric(v) }
func SigmoidBwd(dst, grad, y []float32)          { sigmoidBwdGeneric(dst, grad, y) }
func TanhFwd(dst, src []float32)                 { tanhFwdGeneric(dst, src) }
func TanhBwd(dst, grad, y []float32)             { tanhBwdGeneric(dst, grad, y) }
func ExpShift(dst, src []float32, shift float32) { expShiftGeneric(dst, src, shift) }
func AddSlice(dst, src []float32)                { addSliceGeneric(dst, src) }

func AddSlices(dst, x, y []float32)        { addSlicesGeneric(dst, x, y) }
func SubSlices(dst, x, y []float32)        { subSlicesGeneric(dst, x, y) }
func MulSlices(dst, x, y []float32)        { mulSlicesGeneric(dst, x, y) }
func DivSlices(dst, x, y []float32)        { divSlicesGeneric(dst, x, y) }
func ScaleSlice(dst []float32, s float32)  { scaleSliceGeneric(dst, s) }
func SoftmaxBwdAdd(dst, grad, y []float32) { softmaxBwdAddGeneric(dst, grad, y) }
func AdamStep(w, g, m, v []float32, beta1, beta2, rc1, rc2, lr, eps, wd float32) {
	adamStepGeneric(w, g, m, v, beta1, beta2, rc1, rc2, lr, eps, wd)
}
func SGDStep(w, g, vel []float32, momentum, lr float32) {
	sgdStepGeneric(w, g, vel, momentum, lr)
}

func GeluFwd(dst, src []float32)       { geluFwdGeneric(dst, src) }
func GeluBwd(dst, grad, src []float32) { geluBwdGeneric(dst, grad, src) }
func LnFwdRow(out, xhat, src, gamma, beta []float32, eps float32) float32 {
	return lnFwdRowGeneric(out, xhat, src, gamma, beta, eps)
}
func LnBwdRow(out, g, xhat, gamma, gradGamma, gradBeta []float32, invStd float32) {
	lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta, invStd)
}

func DotVec(a, b []float32) float32  { return dotVecGeneric(a, b) }
func Axpy(a float32, x, y []float32) { axpyGeneric(a, x, y) }

func DotVecs(qs, k []float32, out []float32) { dotVecsGeneric(qs, k, out) }

func Axpys(ws []float32, v, outs []float32) { axpysGeneric(ws, v, outs) }

// AxpyRows accumulates out += sum over i of ws[i] * rows[i][off:off+d].
func AxpyRows(out, ws []float32, rows [][]float32, off int) {
	for i, w := range ws {
		axpyGeneric(w, rows[i][off:off+len(out)], out)
	}
}
