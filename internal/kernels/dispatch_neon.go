//go:build goexperiment.simd && arm64 && go1.27

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
// Every kernel of the amd64 file has its NEON form here: the ops a
// decode step runs per token (the attention dot products and value
// accumulation, the exponentials, the element-wise rows) and the
// training-side kernels (activations and their gradients, the optimizer
// steps, LayerNorm), plus the Hadamard rotation and the delta-rule reads.

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

// hsum4 reduces a vector accumulator to a scalar, lanes in index order.
func hsum4(v archsimd.Float32x4) float32 {
	var tmp [4]float32
	simd.StoreF32x4(v, tmp[:])
	return tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

func ReluFwd(dst, src []float32) {
	zero := archsimd.BroadcastFloat32x4(0)
	map4(dst, src, func(v archsimd.Float32x4) archsimd.Float32x4 { return v.Max(zero) })
}

func ReluBwd(dst, grad, src []float32) {
	zero := archsimd.BroadcastFloat32x4(0)
	map4x2(dst, grad, src, func(g, v archsimd.Float32x4) archsimd.Float32x4 {
		return g.Masked(v.Greater(zero))
	})
}

func LeakyFwd(dst, src []float32, alpha float32) {
	av := archsimd.BroadcastFloat32x4(alpha)
	// max(x, alpha*x) equals leaky ReLU for alpha in [0, 1).
	map4(dst, src, func(v archsimd.Float32x4) archsimd.Float32x4 { return v.Max(v.Mul(av)) })
}

func LeakyBwd(dst, grad, src []float32, alpha float32) {
	zero := archsimd.BroadcastFloat32x4(0)
	av := archsimd.BroadcastFloat32x4(alpha)
	map4x2(dst, grad, src, func(g, v archsimd.Float32x4) archsimd.Float32x4 {
		ga := g.Mul(av)
		return ga.Add(g.Sub(ga).Masked(v.Greater(zero)))
	})
}

// GeluMul is Gemma's gate: gelu(gate) * up, in place on gate. The tanh
// approximation the trained models use rewrites as a sigmoid, so this is
// SiluMul with the argument run through the cubic first.
func GeluMul(gate, up []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	inner := archsimd.BroadcastFloat32x4(geluTanhInner)
	cube := archsimd.BroadcastFloat32x4(geluTanhCube)
	map4x2(gate, gate, up, func(g, u archsimd.Float32x4) archsimd.Float32x4 {
		y := inner.Mul(g.Add(cube.Mul(g).Mul(g).Mul(g)))
		return g.Div(one.Add(vexpf4(y.Neg()))).Mul(u)
	})
}

func SigmoidBwd(dst, grad, y []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	map4x2(dst, grad, y, func(g, yv archsimd.Float32x4) archsimd.Float32x4 {
		return g.Mul(yv).Mul(one.Sub(yv))
	})
}

func TanhFwd(dst, src []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	map4(dst, src, func(v archsimd.Float32x4) archsimd.Float32x4 {
		e2 := vexpf4(v.Add(v))
		return e2.Sub(one).Div(e2.Add(one))
	})
}

func TanhBwd(dst, grad, y []float32) {
	one := archsimd.BroadcastFloat32x4(1)
	map4x2(dst, grad, y, func(g, yv archsimd.Float32x4) archsimd.Float32x4 {
		return g.Mul(one.Sub(yv.Mul(yv)))
	})
}

func SoftmaxBwdAdd(dst, grad, y []float32) {
	if len(y) < 8 {
		softmaxBwdAddGeneric(dst, grad, y)
		return
	}
	n := len(y) &^ 3
	var acc archsimd.Float32x4
	for i := 0; i < n; i += 4 {
		acc = simd.LoadF32x4(grad[i:]).MulAdd(simd.LoadF32x4(y[i:]), acc)
	}
	dot := hsum4(acc)
	for i := n; i < len(y); i++ {
		dot += grad[i] * y[i]
	}
	dv := archsimd.BroadcastFloat32x4(dot)
	for i := 0; i < n; i += 4 {
		gv := simd.LoadF32x4(grad[i:]).Sub(dv)
		simd.StoreF32x4(simd.LoadF32x4(y[i:]).MulAdd(gv, simd.LoadF32x4(dst[i:])), dst[i:])
	}
	for i := n; i < len(y); i++ {
		dst[i] += y[i] * (grad[i] - dot)
	}
}

func AdamStep(w, g, m, v []float32, beta1, beta2, rc1, rc2, lr, eps, wd float32) {
	b1 := archsimd.BroadcastFloat32x4(beta1)
	ib1 := archsimd.BroadcastFloat32x4(1 - beta1)
	b2 := archsimd.BroadcastFloat32x4(beta2)
	ib2 := archsimd.BroadcastFloat32x4(1 - beta2)
	c1 := archsimd.BroadcastFloat32x4(rc1)
	c2 := archsimd.BroadcastFloat32x4(rc2)
	lrv := archsimd.BroadcastFloat32x4(lr)
	epsv := archsimd.BroadcastFloat32x4(eps)
	wdv := archsimd.BroadcastFloat32x4(wd)
	step := func(wv, gv, mv, vv archsimd.Float32x4) (archsimd.Float32x4, archsimd.Float32x4, archsimd.Float32x4) {
		mv = gv.MulAdd(ib1, mv.Mul(b1))
		vv = gv.Mul(gv).MulAdd(ib2, vv.Mul(b2))
		update := mv.Mul(c1).Div(vv.Mul(c2).Sqrt().Add(epsv))
		update = wv.MulAdd(wdv, update)
		return wv.Sub(update.Mul(lrv)), mv, vv
	}
	for len(w) >= 4 {
		wv, mv, vv := step(simd.LoadF32x4(w), simd.LoadF32x4(g), simd.LoadF32x4(m), simd.LoadF32x4(v))
		simd.StoreF32x4(wv, w)
		simd.StoreF32x4(mv, m)
		simd.StoreF32x4(vv, v)
		w, g, m, v = w[4:], g[4:], m[4:], v[4:]
	}
	if len(w) > 0 {
		wv, mv, vv := step(simd.LoadF32x4Part(w), simd.LoadF32x4Part(g), simd.LoadF32x4Part(m), simd.LoadF32x4Part(v))
		simd.StoreF32x4Part(wv, w)
		simd.StoreF32x4Part(mv, m)
		simd.StoreF32x4Part(vv, v)
	}
}

func SGDStep(w, g, vel []float32, momentum, lr float32) {
	mo := archsimd.BroadcastFloat32x4(momentum)
	nlr := archsimd.BroadcastFloat32x4(-lr)
	for len(w) >= 4 {
		vv := simd.LoadF32x4(g).MulAdd(nlr, simd.LoadF32x4(vel).Mul(mo))
		simd.StoreF32x4(vv, vel)
		simd.StoreF32x4(simd.LoadF32x4(w).Add(vv), w)
		w, g, vel = w[4:], g[4:], vel[4:]
	}
	if len(w) > 0 {
		vv := simd.LoadF32x4Part(g).MulAdd(nlr, simd.LoadF32x4Part(vel).Mul(mo))
		simd.StoreF32x4Part(vv, vel)
		simd.StoreF32x4Part(simd.LoadF32x4Part(w).Add(vv), w)
	}
}

// verf4 computes erf(x) per lane with the Abramowitz-Stegun 7.1.26
// polynomial, the same one as the AVX2 verf; symmetry handles negative
// inputs through the sign bit.
func verf4(x archsimd.Float32x4) archsimd.Float32x4 {
	signBit := archsimd.BroadcastUint32x4(0x80000000)
	sign := x.ToBits().And(signBit)
	ax := x.Abs()

	one := archsimd.BroadcastFloat32x4(1)
	t := one.Div(archsimd.BroadcastFloat32x4(0.3275911).MulAdd(ax, one))
	p := archsimd.BroadcastFloat32x4(1.061405429)
	p = p.MulAdd(t, archsimd.BroadcastFloat32x4(-1.453152027))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x4(1.421413741))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x4(-0.284496736))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x4(0.254829592))
	p = p.Mul(t)
	e := vexpf4(ax.Mul(ax).Neg())
	erfAbs := one.Sub(p.Mul(e))
	return erfAbs.ToBits().Or(sign).BitsToFloat32()
}

func GeluFwd(dst, src []float32) {
	half := archsimd.BroadcastFloat32x4(0.5)
	one := archsimd.BroadcastFloat32x4(1)
	invSqrt2 := archsimd.BroadcastFloat32x4(0.7071067811865476)
	map4(dst, src, func(v archsimd.Float32x4) archsimd.Float32x4 {
		return half.Mul(v).Mul(one.Add(verf4(v.Mul(invSqrt2))))
	})
}

func GeluBwd(dst, grad, src []float32) {
	half := archsimd.BroadcastFloat32x4(0.5)
	one := archsimd.BroadcastFloat32x4(1)
	invSqrt2 := archsimd.BroadcastFloat32x4(0.7071067811865476)
	invSqrt2Pi := archsimd.BroadcastFloat32x4(0.3989422804014327)
	negHalf := archsimd.BroadcastFloat32x4(-0.5)
	map4x2(dst, grad, src, func(g, v archsimd.Float32x4) archsimd.Float32x4 {
		cdf := half.Mul(one.Add(verf4(v.Mul(invSqrt2))))
		pdf := v.Mul(invSqrt2Pi).Mul(vexpf4(negHalf.Mul(v).Mul(v)))
		return g.Mul(cdf.Add(pdf))
	})
}

func LnFwdRow(out, xhat, src, gamma, beta []float32, eps float32) float32 {
	n := float32(len(src))

	var acc archsimd.Float32x4
	s := src
	for len(s) >= 4 {
		acc = acc.Add(simd.LoadF32x4(s))
		s = s[4:]
	}
	if len(s) > 0 {
		acc = acc.Add(simd.LoadF32x4Part(s)) // missing lanes are zero
	}
	mean := hsum4(acc) / n

	meanV := archsimd.BroadcastFloat32x4(mean)
	var vacc archsimd.Float32x4
	s = src
	for len(s) >= 4 {
		d := simd.LoadF32x4(s).Sub(meanV)
		vacc = d.MulAdd(d, vacc)
		s = s[4:]
	}
	// Scalar tail: zero-filled part-load lanes would skew (v-mean)^2.
	variance := hsum4(vacc)
	for _, v := range s {
		d := v - mean
		variance += d * d
	}
	variance /= n
	invStd := 1 / SqrtF(variance+eps)

	invStdV := archsimd.BroadcastFloat32x4(invStd)
	o, xh, sr, ga, be := out, xhat, src, gamma, beta
	for len(sr) >= 4 {
		h := simd.LoadF32x4(sr).Sub(meanV).Mul(invStdV)
		simd.StoreF32x4(h, xh)
		simd.StoreF32x4(h.MulAdd(simd.LoadF32x4(ga), simd.LoadF32x4(be)), o)
		o, xh, sr, ga, be = o[4:], xh[4:], sr[4:], ga[4:], be[4:]
	}
	if len(sr) > 0 {
		h := simd.LoadF32x4Part(sr).Sub(meanV).Mul(invStdV)
		simd.StoreF32x4Part(h, xh)
		simd.StoreF32x4Part(h.MulAdd(simd.LoadF32x4Part(ga), simd.LoadF32x4Part(be)), o)
	}
	return invStd
}

func LnBwdRow(out, g, xhat, gamma, gradGamma, gradBeta []float32, invStd float32) {
	n := float32(len(g))

	var acc1, acc2 archsimd.Float32x4
	gs, xs, gas, ggs, gbs := g, xhat, gamma, gradGamma, gradBeta
	for len(gs) >= 4 {
		gv := simd.LoadF32x4(gs)
		xh := simd.LoadF32x4(xs)
		simd.StoreF32x4(gv.MulAdd(xh, simd.LoadF32x4(ggs)), ggs)
		simd.StoreF32x4(gv.Add(simd.LoadF32x4(gbs)), gbs)
		dx := gv.Mul(simd.LoadF32x4(gas))
		acc1 = acc1.Add(dx)
		acc2 = dx.MulAdd(xh, acc2)
		gs, xs, gas, ggs, gbs = gs[4:], xs[4:], gas[4:], ggs[4:], gbs[4:]
	}
	if len(gs) > 0 {
		gv := simd.LoadF32x4Part(gs)
		xh := simd.LoadF32x4Part(xs)
		simd.StoreF32x4Part(gv.MulAdd(xh, simd.LoadF32x4Part(ggs)), ggs)
		simd.StoreF32x4Part(gv.Add(simd.LoadF32x4Part(gbs)), gbs)
		dx := gv.Mul(simd.LoadF32x4Part(gas))
		acc1 = acc1.Add(dx)
		acc2 = dx.MulAdd(xh, acc2)
	}
	sumDXhat := hsum4(acc1)
	sumDXhatXhat := hsum4(acc2)

	kV := archsimd.BroadcastFloat32x4(invStd / n)
	nV := archsimd.BroadcastFloat32x4(n)
	s1V := archsimd.BroadcastFloat32x4(sumDXhat)
	s2V := archsimd.BroadcastFloat32x4(sumDXhatXhat)
	o := out
	gs, xs, gas = g, xhat, gamma
	for len(gs) >= 4 {
		dx := simd.LoadF32x4(gs).Mul(simd.LoadF32x4(gas))
		t := dx.Mul(nV).Sub(s1V).Sub(simd.LoadF32x4(xs).Mul(s2V))
		simd.StoreF32x4(t.Mul(kV), o)
		o, gs, xs, gas = o[4:], gs[4:], xs[4:], gas[4:]
	}
	if len(gs) > 0 {
		dx := simd.LoadF32x4Part(gs).Mul(simd.LoadF32x4Part(gas))
		t := dx.Mul(nV).Sub(s1V).Sub(simd.LoadF32x4Part(xs).Mul(s2V))
		simd.StoreF32x4Part(t.Mul(kV), o)
	}
}

// Hadamard is the Walsh-Hadamard transform of v in place, times scale,
// Sylvester order, for a power-of-two length. The two stages within a
// lane group of four go through the scalar butterfly; every stage from a
// stride of four up is a vector add and subtract across whole lanes, two
// stages per radix-4 pass, with the scale riding on whichever pass is
// last.
func Hadamard(v []float32, scale float32) {
	n := len(v)
	if n < 8 {
		hadamardGeneric(v, scale)
		return
	}
	sv := archsimd.BroadcastFloat32x4(scale)
	for i := 0; i+4 <= n; i += 4 {
		b := v[i : i+4 : i+4]
		a0, a1, a2, a3 := b[0], b[1], b[2], b[3]
		a0, a1 = a0+a1, a0-a1
		a2, a3 = a2+a3, a2-a3
		b[0], b[2] = a0+a2, a0-a2
		b[1], b[3] = a1+a3, a1-a3
	}
	hEnd := 4
	for 4*hEnd <= n {
		hEnd *= 4
	}
	tail := 2*hEnd <= n
	h := 4
	for ; 4*h <= n; h *= 4 {
		s := archsimd.BroadcastFloat32x4(1)
		if !tail && 4*h == hEnd {
			s = sv
		}
		for i := 0; i < n; i += 4 * h {
			for j := i; j < i+h; j += 4 {
				a := simd.LoadF32x4(v[j:])
				b := simd.LoadF32x4(v[j+h:])
				c := simd.LoadF32x4(v[j+2*h:])
				d := simd.LoadF32x4(v[j+3*h:])
				ab, abd := a.Add(b), a.Sub(b)
				cd, cdd := c.Add(d), c.Sub(d)
				simd.StoreF32x4(ab.Add(cd).Mul(s), v[j:])
				simd.StoreF32x4(abd.Add(cdd).Mul(s), v[j+h:])
				simd.StoreF32x4(ab.Sub(cd).Mul(s), v[j+2*h:])
				simd.StoreF32x4(abd.Sub(cdd).Mul(s), v[j+3*h:])
			}
		}
	}
	if tail {
		for j := 0; j < h; j += 4 {
			x := simd.LoadF32x4(v[j:])
			y := simd.LoadF32x4(v[j+h:])
			simd.StoreF32x4(x.Add(y).Mul(sv), v[j:])
			simd.StoreF32x4(x.Sub(y).Mul(sv), v[j+h:])
		}
	}
}

// DecayRead scales row by decay in place and adds k times the scaled
// row into mem, one read and one write of the row.
func DecayRead(row []float32, decay, k float32, mem []float32) {
	if len(row) < 8 {
		decayReadGeneric(row, decay, k, mem)
		return
	}
	dv := archsimd.BroadcastFloat32x4(decay)
	kv := archsimd.BroadcastFloat32x4(k)
	n := len(row) &^ 3
	for i := 0; i < n; i += 4 {
		v := simd.LoadF32x4(row[i:]).Mul(dv)
		simd.StoreF32x4(v, row[i:])
		simd.StoreF32x4(v.MulAdd(kv, simd.LoadF32x4(mem[i:])), mem[i:])
	}
	for i := n; i < len(row); i++ {
		v := row[i] * decay
		row[i] = v
		mem[i] += k * v
	}
}

// WriteRead adds k times delta into row in place and q times the
// updated row into out, one read and one write of the row.
func WriteRead(row, delta []float32, k, q float32, out []float32) {
	if len(row) < 8 {
		writeReadGeneric(row, delta, k, q, out)
		return
	}
	kv := archsimd.BroadcastFloat32x4(k)
	qv := archsimd.BroadcastFloat32x4(q)
	n := len(row) &^ 3
	for i := 0; i < n; i += 4 {
		v := simd.LoadF32x4(delta[i:]).MulAdd(kv, simd.LoadF32x4(row[i:]))
		simd.StoreF32x4(v, row[i:])
		simd.StoreF32x4(v.MulAdd(qv, simd.LoadF32x4(out[i:])), out[i:])
	}
	for i := n; i < len(row); i++ {
		v := row[i] + k*delta[i]
		row[i] = v
		out[i] += q * v
	}
}
