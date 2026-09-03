//go:build goexperiment.simd && arm64

package kernels

import (
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// 128-bit NEON kernels, the arm64 half of dispatch_simd.go. Four lanes
// where AVX2 has eight, and no dot-product instruction in the exposed
// API, so the integer matvecs widen by hand (see quant/quant_neon.go);
// the float work here maps straight across, FMLA for MulAdd included.
//
// Vectorized: the ops a decode step runs per token — the attention dot
// products and value accumulation, the exponentials, the element-wise
// rows. The training-side kernels keep the portable bodies; they run once
// per step over long slices where the scalar loop is not the wall.

// vexpf4 is vexpf's polynomial at four lanes: range reduction to
// r in [-ln2/2, ln2/2] with a split ln2, a degree-5 minimax on r, and
// the 2^n scale folded into the exponent bits. Same constants as the
// AVX2 one, so the two agree to the last bit on the same input.
func vexpf4(x archsimd.Float32x4) archsimd.Float32x4 {
	x = x.Min(archsimd.BroadcastFloat32x4(88.0)).Max(archsimd.BroadcastFloat32x4(-87.0))
	z := x.Mul(archsimd.BroadcastFloat32x4(1.44269504)).Round()
	r := z.MulAdd(archsimd.BroadcastFloat32x4(-0.693359375), x)
	r = z.MulAdd(archsimd.BroadcastFloat32x4(2.12194440e-4), r)
	p := archsimd.BroadcastFloat32x4(1.9875691500e-4)
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.3981999507e-3))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(8.3334519073e-3))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(4.1665795894e-2))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(1.6666665459e-1))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x4(5.0000001201e-1))
	er := p.MulAdd(r.Mul(r), r).Add(archsimd.BroadcastFloat32x4(1))
	// 2^n through the exponent field. arm64 has no int-to-float bit cast,
	// so the biased exponent is built in the float domain first: the input
	// clamp keeps z in [-126, 127], so z+127 is a positive integer value
	// and the unsigned convert is exact.
	scale := z.Add(archsimd.BroadcastFloat32x4(127)).ConvertToUint32().ShiftAllLeft(23).BitsToFloat32()
	return er.Mul(scale)
}

// map4 runs a four-lane body over dst/src, the tail through a partial
// load and store so it takes the same path as the full vectors; the amd64
// twin does the same, which is what keeps the two architectures agreeing
// on the last elements of an odd-length row.
func map4(dst, src []float32, f func(archsimd.Float32x4) archsimd.Float32x4) {
	for len(dst) >= 4 && len(src) >= 4 {
		simd.StoreF32x4(f(simd.LoadF32x4(src)), dst)
		dst, src = dst[4:], src[4:]
	}
	if len(dst) > 0 {
		simd.StoreF32x4Part(f(simd.LoadF32x4Part(src)), dst)
	}
}

// map4x2 is map4 over two inputs.
func map4x2(dst, x, y []float32, f func(a, b archsimd.Float32x4) archsimd.Float32x4) {
	for len(dst) >= 4 && len(x) >= 4 && len(y) >= 4 {
		simd.StoreF32x4(f(simd.LoadF32x4(x), simd.LoadF32x4(y)), dst)
		dst, x, y = dst[4:], x[4:], y[4:]
	}
	if len(dst) > 0 {
		simd.StoreF32x4Part(f(simd.LoadF32x4Part(x), simd.LoadF32x4Part(y)), dst)
	}
}

func ExpShift(dst, src []float32, shift float32) {
	sv := archsimd.BroadcastFloat32x4(shift)
	map4(dst, src, func(v archsimd.Float32x4) archsimd.Float32x4 {
		return vexpf4(v.Sub(sv))
	})
}

// sigmoid4 is 1/(1+e^-x), the gate silu multiplies by.
func sigmoid4(v archsimd.Float32x4) archsimd.Float32x4 {
	one := archsimd.BroadcastFloat32x4(1)
	return one.Div(one.Add(vexpf4(v.Neg())))
}

func SigmoidFwd(dst, src []float32) {
	map4(dst, src, sigmoid4)
}

func Silu(v []float32) {
	map4(v, v, func(x archsimd.Float32x4) archsimd.Float32x4 {
		return x.Mul(sigmoid4(x))
	})
}

// SiluMul is the SwiGLU gate: gate = silu(gate) * up.
func SiluMul(gate, up []float32) {
	map4x2(gate, gate, up, func(g, u archsimd.Float32x4) archsimd.Float32x4 {
		return g.Mul(sigmoid4(g)).Mul(u)
	})
}

// AddSlice accumulates dst += src.
func AddSlice(dst, src []float32) {
	map4x2(dst, dst, src, func(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Add(b) })
}

func AddSlices(dst, x, y []float32) {
	map4x2(dst, x, y, func(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Add(b) })
}

func SubSlices(dst, x, y []float32) {
	map4x2(dst, x, y, func(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Sub(b) })
}

func MulSlices(dst, x, y []float32) {
	map4x2(dst, x, y, func(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Mul(b) })
}

func DivSlices(dst, x, y []float32) {
	map4x2(dst, x, y, func(a, b archsimd.Float32x4) archsimd.Float32x4 { return a.Div(b) })
}

func ScaleSlice(dst []float32, s float32) {
	sv := archsimd.BroadcastFloat32x4(s)
	map4(dst, dst, func(v archsimd.Float32x4) archsimd.Float32x4 { return v.Mul(sv) })
}

// DotVec sums a*b four lanes at a time, folding the lanes in index order
// so the result is stable across runs.
func DotVec(a, b []float32) float32 {
	if len(a) < 8 {
		return dotVecGeneric(a, b)
	}
	n := min(len(a), len(b)) &^ 3
	var acc archsimd.Float32x4
	for i := 0; i < n; i += 4 {
		acc = simd.LoadF32x4(a[i:]).MulAdd(simd.LoadF32x4(b[i:]), acc)
	}
	var buf [4]float32
	simd.StoreF32x4(acc, buf[:])
	s := buf[0] + buf[1] + buf[2] + buf[3]
	for i := n; i < len(a) && i < len(b); i++ {
		s += a[i] * b[i]
	}
	return s
}

// Axpy computes y += a*x, four lanes at a time.
func Axpy(a float32, x, y []float32) {
	if len(x) < 8 {
		axpyGeneric(a, x, y)
		return
	}
	av := archsimd.BroadcastFloat32x4(a)
	n := min(len(x), len(y)) &^ 3
	for i := 0; i < n; i += 4 {
		simd.StoreF32x4(simd.LoadF32x4(x[i:]).MulAdd(av, simd.LoadF32x4(y[i:])), y[i:])
	}
	for i := n; i < len(x) && i < len(y); i++ {
		y[i] += a * x[i]
	}
}

// AxpyRows accumulates out += sum over i of ws[i] * rows[i][off:off+d].
// The output block stays in registers across every row, which is what
// makes it worth having over a call per row; see the amd64 twin.
func AxpyRows(out, ws []float32, rows [][]float32, off int) {
	d := len(out)
	if d < 8 {
		for i, w := range ws {
			axpyGeneric(w, rows[i][off:off+d], out)
		}
		return
	}
	n := d &^ 3
	for b := 0; b+32 <= n; b += 32 {
		o := out[b:]
		a0, a1, a2, a3 := simd.LoadF32x4(o), simd.LoadF32x4(o[4:]), simd.LoadF32x4(o[8:]), simd.LoadF32x4(o[12:])
		a4, a5, a6, a7 := simd.LoadF32x4(o[16:]), simd.LoadF32x4(o[20:]), simd.LoadF32x4(o[24:]), simd.LoadF32x4(o[28:])
		for i, w := range ws {
			av := archsimd.BroadcastFloat32x4(w)
			r := rows[i][off+b:]
			a0 = simd.LoadF32x4(r).MulAdd(av, a0)
			a1 = simd.LoadF32x4(r[4:]).MulAdd(av, a1)
			a2 = simd.LoadF32x4(r[8:]).MulAdd(av, a2)
			a3 = simd.LoadF32x4(r[12:]).MulAdd(av, a3)
			a4 = simd.LoadF32x4(r[16:]).MulAdd(av, a4)
			a5 = simd.LoadF32x4(r[20:]).MulAdd(av, a5)
			a6 = simd.LoadF32x4(r[24:]).MulAdd(av, a6)
			a7 = simd.LoadF32x4(r[28:]).MulAdd(av, a7)
		}
		simd.StoreF32x4(a0, o)
		simd.StoreF32x4(a1, o[4:])
		simd.StoreF32x4(a2, o[8:])
		simd.StoreF32x4(a3, o[12:])
		simd.StoreF32x4(a4, o[16:])
		simd.StoreF32x4(a5, o[20:])
		simd.StoreF32x4(a6, o[24:])
		simd.StoreF32x4(a7, o[28:])
	}
	for b := n &^ 31; b < n; b += 4 {
		o := out[b:]
		acc := simd.LoadF32x4(o)
		for i, w := range ws {
			acc = simd.LoadF32x4(rows[i][off+b:]).MulAdd(archsimd.BroadcastFloat32x4(w), acc)
		}
		simd.StoreF32x4(acc, o)
	}
	for i, w := range ws {
		r := rows[i][off:]
		for k := n; k < d; k++ {
			out[k] += w * r[k]
		}
	}
}

// DotVecs is DotVec for a stack of queries against one key row.
func DotVecs(qs, k []float32, out []float32) {
	d := len(k)
	for i := range out {
		out[i] = DotVec(qs[i*d:(i+1)*d], k)
	}
}

// Axpys accumulates one value row into several outputs, one weight each.
func Axpys(ws []float32, v, outs []float32) {
	d := len(v)
	for i, w := range ws {
		Axpy(w, v, outs[i*d:(i+1)*d])
	}
}

// The rest keep the portable bodies: they are training-side kernels that
// run once per step over long slices, not per token.

func ReluFwd(dst, src []float32)                 { reluFwdGeneric(dst, src) }
func ReluBwd(dst, grad, src []float32)           { reluBwdGeneric(dst, grad, src) }
func LeakyFwd(dst, src []float32, alpha float32) { leakyFwdGeneric(dst, src, alpha) }
func LeakyBwd(dst, grad, src []float32, alpha float32) {
	leakyBwdGeneric(dst, grad, src, alpha)
}
func GeluMul(gate, up []float32)        { geluMulGeneric(gate, up) }
func SigmoidBwd(dst, grad, y []float32) { sigmoidBwdGeneric(dst, grad, y) }
func TanhFwd(dst, src []float32)        { tanhFwdGeneric(dst, src) }
func TanhBwd(dst, grad, y []float32)    { tanhBwdGeneric(dst, grad, y) }
func SoftmaxBwdAdd(dst, grad, y []float32) {
	softmaxBwdAddGeneric(dst, grad, y)
}
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
