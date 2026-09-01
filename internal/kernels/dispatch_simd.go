//go:build goexperiment.simd && amd64

package kernels

import (
	"simd/archsimd"

	"github.com/mattn/tensai/internal/simd"
)

// AVX2 versions of the element-wise kernels. Bodies stay free of scalar
// float instructions (SSE encodings) while the vector upper state is dirty
// — see dot_simd.go — and every kernel clears that state before returning.

// vexpf computes e^x per lane with a Cephes-style degree-5 polynomial after
// range reduction x = z*ln2 + r. Inputs are clamped to about [-87, 88], so
// the result saturates instead of overflowing to Inf; relative error is a
// few float32 ulps, far below what training precision requires.
func vexpf(x archsimd.Float32x8) archsimd.Float32x8 {
	x = x.Min(archsimd.BroadcastFloat32x8(88.0)).Max(archsimd.BroadcastFloat32x8(-87.0))
	z := simd.RoundEven(x.Mul(archsimd.BroadcastFloat32x8(1.44269504)))
	n := z.ConvertToInt32() // exact: z is already an integer value
	// r = x - z*ln2, with ln2 split for extra precision (Cody-Waite).
	r := z.MulAdd(archsimd.BroadcastFloat32x8(-0.693359375), x)
	r = z.MulAdd(archsimd.BroadcastFloat32x8(2.12194440e-4), r)
	p := archsimd.BroadcastFloat32x8(1.9875691500e-4)
	p = p.MulAdd(r, archsimd.BroadcastFloat32x8(1.3981999507e-3))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x8(8.3334519073e-3))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x8(4.1665795894e-2))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x8(1.6666665459e-1))
	p = p.MulAdd(r, archsimd.BroadcastFloat32x8(5.0000001201e-1))
	// e^r = 1 + r + r^2 * p(r)
	er := p.MulAdd(r.Mul(r), r).Add(archsimd.BroadcastFloat32x8(1))
	// scale by 2^n via the exponent bits
	scale := n.Add(archsimd.BroadcastInt32x8(127)).ShiftAllLeft(23).AsFloat32x8()
	return er.Mul(scale)
}

// mapSlices runs an 8-lane vector body over dst/src with a masked tail.
func mapSlices(dst, src []float32, f func(archsimd.Float32x8) archsimd.Float32x8) {
	for len(dst) >= 8 && len(src) >= 8 {
		simd.StoreF32x8(f(simd.LoadF32x8(src)), dst)
		dst, src = dst[8:], src[8:]
	}
	if len(dst) > 0 {
		simd.StoreF32x8Part(f(simd.LoadF32x8Part(src)), dst)
	}
	archsimd.ClearAVXUpperBits()
}

// mapSlices2 is mapSlices over two inputs.
func mapSlices2(dst, x, y []float32, f func(a, b archsimd.Float32x8) archsimd.Float32x8) {
	for len(dst) >= 8 && len(x) >= 8 && len(y) >= 8 {
		simd.StoreF32x8(f(simd.LoadF32x8(x), simd.LoadF32x8(y)), dst)
		dst, x, y = dst[8:], x[8:], y[8:]
	}
	if len(dst) > 0 {
		simd.StoreF32x8Part(f(simd.LoadF32x8Part(x), simd.LoadF32x8Part(y)), dst)
	}
	archsimd.ClearAVXUpperBits()
}

func ReluFwd(dst, src []float32) {
	if !simd.HasAVX2 {
		reluFwdGeneric(dst, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Max(zero)
	})
}

func ReluBwd(dst, grad, src []float32) {
	if !simd.HasAVX2 {
		reluBwdGeneric(dst, grad, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices2(dst, grad, src, func(g, v archsimd.Float32x8) archsimd.Float32x8 {
		return g.AsInt32x8().And(v.Greater(zero).ToInt32x8()).AsFloat32x8()
	})
}

func LeakyFwd(dst, src []float32, alpha float32) {
	if !simd.HasAVX2 {
		leakyFwdGeneric(dst, src, alpha)
		return
	}
	av := archsimd.BroadcastFloat32x8(alpha)
	// max(x, alpha*x) equals leaky ReLU for alpha in [0, 1).
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Max(v.Mul(av))
	})
}

func LeakyBwd(dst, grad, src []float32, alpha float32) {
	if !simd.HasAVX2 {
		leakyBwdGeneric(dst, grad, src, alpha)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	av := archsimd.BroadcastFloat32x8(alpha)
	mapSlices2(dst, grad, src, func(g, v archsimd.Float32x8) archsimd.Float32x8 {
		ga := g.Mul(av)
		rest := g.Sub(ga).AsInt32x8().And(v.Greater(zero).ToInt32x8()).AsFloat32x8()
		return ga.Add(rest)
	})
}

func SigmoidFwd(dst, src []float32) {
	if !simd.HasAVX2 {
		sigmoidFwdGeneric(dst, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return one.Div(one.Add(vexpf(zero.Sub(v))))
	})
}

// SiluMul computes gate = gate * sigmoid(gate) * up — the SwiGLU gate,
// a transformer decode's only transcendental hot spot.
func SiluMul(gate, up []float32) {
	if !simd.HasAVX2 {
		siluMulGeneric(gate, up)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices2(gate, gate, up, func(g, u archsimd.Float32x8) archsimd.Float32x8 {
		return g.Div(one.Add(vexpf(zero.Sub(g)))).Mul(u)
	})
}

// GeluMul is Gemma's gate: gelu(gate) * up, in place on gate. The tanh
// approximation the trained models use rewrites as a sigmoid, so this is
// SiluMul with the argument run through the cubic first.
func GeluMul(gate, up []float32) {
	if !simd.HasAVX2 {
		geluMulGeneric(gate, up)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	zero := archsimd.BroadcastFloat32x8(0)
	inner := archsimd.BroadcastFloat32x8(geluTanhInner)
	cube := archsimd.BroadcastFloat32x8(geluTanhCube)
	mapSlices2(gate, gate, up, func(g, u archsimd.Float32x8) archsimd.Float32x8 {
		y := inner.Mul(g.Add(cube.Mul(g).Mul(g).Mul(g)))
		return g.Div(one.Add(vexpf(zero.Sub(y)))).Mul(u)
	})
}

// Silu applies x * sigmoid(x) in place. SiluMul is the same thing paired
// with a gate; the delta rule wants it on its own, over a convolution's
// output rather than a SwiGLU's.
func Silu(v []float32) {
	if !simd.HasAVX2 {
		siluGeneric(v)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices(v, v, func(x archsimd.Float32x8) archsimd.Float32x8 {
		return x.Div(one.Add(vexpf(zero.Sub(x))))
	})
}

func SigmoidBwd(dst, grad, y []float32) {
	if !simd.HasAVX2 {
		sigmoidBwdGeneric(dst, grad, y)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices2(dst, grad, y, func(g, yv archsimd.Float32x8) archsimd.Float32x8 {
		return g.Mul(yv).Mul(one.Sub(yv))
	})
}

func TanhFwd(dst, src []float32) {
	if !simd.HasAVX2 {
		tanhFwdGeneric(dst, src)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		e2 := vexpf(v.Add(v))
		return e2.Sub(one).Div(e2.Add(one))
	})
}

func TanhBwd(dst, grad, y []float32) {
	if !simd.HasAVX2 {
		tanhBwdGeneric(dst, grad, y)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices2(dst, grad, y, func(g, yv archsimd.Float32x8) archsimd.Float32x8 {
		return g.Mul(one.Sub(yv.Mul(yv)))
	})
}

func ExpShift(dst, src []float32, shift float32) {
	if !simd.HasAVX2 {
		expShiftGeneric(dst, src, shift)
		return
	}
	sv := archsimd.BroadcastFloat32x8(shift)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return vexpf(v.Sub(sv))
	})
}

// AddSlices, SubSlices, MulSlices and DivSlices compute dst = x op y, the
// element-wise ops the tensor package broadcasts with. mapSlices2 already
// takes a separate destination, so each is one vector body.
func AddSlices(dst, x, y []float32) {
	if !simd.HasAVX2 {
		addSlicesGeneric(dst, x, y)
		return
	}
	mapSlices2(dst, x, y, func(a, b archsimd.Float32x8) archsimd.Float32x8 {
		return a.Add(b)
	})
}

func SubSlices(dst, x, y []float32) {
	if !simd.HasAVX2 {
		subSlicesGeneric(dst, x, y)
		return
	}
	mapSlices2(dst, x, y, func(a, b archsimd.Float32x8) archsimd.Float32x8 {
		return a.Sub(b)
	})
}

func MulSlices(dst, x, y []float32) {
	if !simd.HasAVX2 {
		mulSlicesGeneric(dst, x, y)
		return
	}
	mapSlices2(dst, x, y, func(a, b archsimd.Float32x8) archsimd.Float32x8 {
		return a.Mul(b)
	})
}

func DivSlices(dst, x, y []float32) {
	if !simd.HasAVX2 {
		divSlicesGeneric(dst, x, y)
		return
	}
	mapSlices2(dst, x, y, func(a, b archsimd.Float32x8) archsimd.Float32x8 {
		return a.Div(b)
	})
}

func AddSlice(dst, src []float32) {
	if !simd.HasAVX2 {
		addSliceGeneric(dst, src)
		return
	}
	mapSlices2(dst, dst, src, func(d, s archsimd.Float32x8) archsimd.Float32x8 {
		return d.Add(s)
	})
}

func ScaleSlice(dst []float32, s float32) {
	if !simd.HasAVX2 {
		scaleSliceGeneric(dst, s)
		return
	}
	sv := archsimd.BroadcastFloat32x8(s)
	mapSlices(dst, dst, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Mul(sv)
	})
}

func SoftmaxBwdAdd(dst, grad, y []float32) {
	if !simd.HasAVX2 || len(y) < 16 {
		softmaxBwdAddGeneric(dst, grad, y)
		return
	}
	n := len(y) &^ 7
	var acc archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		acc = simd.LoadF32x8(grad[i:]).MulAdd(simd.LoadF32x8(y[i:]), acc)
	}
	dot := hsum(acc)
	for i := n; i < len(y); i++ {
		dot += grad[i] * y[i]
	}
	dv := archsimd.BroadcastFloat32x8(dot)
	for i := 0; i < n; i += 8 {
		gv := simd.LoadF32x8(grad[i:]).Sub(dv)
		simd.StoreF32x8(simd.LoadF32x8(y[i:]).MulAdd(gv, simd.LoadF32x8(dst[i:])), dst[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < len(y); i++ {
		dst[i] += y[i] * (grad[i] - dot)
	}
}

func AdamStep(w, g, m, v []float32, beta1, beta2, rc1, rc2, lr, eps, wd float32) {
	if !simd.HasAVX2 {
		adamStepGeneric(w, g, m, v, beta1, beta2, rc1, rc2, lr, eps, wd)
		return
	}
	b1 := archsimd.BroadcastFloat32x8(beta1)
	ib1 := archsimd.BroadcastFloat32x8(1 - beta1)
	b2 := archsimd.BroadcastFloat32x8(beta2)
	ib2 := archsimd.BroadcastFloat32x8(1 - beta2)
	c1 := archsimd.BroadcastFloat32x8(rc1)
	c2 := archsimd.BroadcastFloat32x8(rc2)
	lrv := archsimd.BroadcastFloat32x8(lr)
	epsv := archsimd.BroadcastFloat32x8(eps)
	wdv := archsimd.BroadcastFloat32x8(wd)
	step := func(wv, gv, mv, vv archsimd.Float32x8) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Float32x8) {
		mv = gv.MulAdd(ib1, mv.Mul(b1))
		vv = gv.Mul(gv).MulAdd(ib2, vv.Mul(b2))
		update := mv.Mul(c1).Div(vv.Mul(c2).Sqrt().Add(epsv))
		update = wv.MulAdd(wdv, update)
		return wv.Sub(update.Mul(lrv)), mv, vv
	}
	for len(w) >= 8 {
		wv, mv, vv := step(simd.LoadF32x8(w), simd.LoadF32x8(g), simd.LoadF32x8(m), simd.LoadF32x8(v))
		simd.StoreF32x8(wv, w)
		simd.StoreF32x8(mv, m)
		simd.StoreF32x8(vv, v)
		w, g, m, v = w[8:], g[8:], m[8:], v[8:]
	}
	if len(w) > 0 {
		wv, mv, vv := step(simd.LoadF32x8Part(w), simd.LoadF32x8Part(g), simd.LoadF32x8Part(m), simd.LoadF32x8Part(v))
		simd.StoreF32x8Part(wv, w)
		simd.StoreF32x8Part(mv, m)
		simd.StoreF32x8Part(vv, v)
	}
	archsimd.ClearAVXUpperBits()
}

func SGDStep(w, g, vel []float32, momentum, lr float32) {
	if !simd.HasAVX2 {
		sgdStepGeneric(w, g, vel, momentum, lr)
		return
	}
	mo := archsimd.BroadcastFloat32x8(momentum)
	nlr := archsimd.BroadcastFloat32x8(-lr)
	for len(w) >= 8 {
		vv := simd.LoadF32x8(g).MulAdd(nlr, simd.LoadF32x8(vel).Mul(mo))
		simd.StoreF32x8(vv, vel)
		simd.StoreF32x8(simd.LoadF32x8(w).Add(vv), w)
		w, g, vel = w[8:], g[8:], vel[8:]
	}
	if len(w) > 0 {
		vv := simd.LoadF32x8Part(g).MulAdd(nlr, simd.LoadF32x8Part(vel).Mul(mo))
		simd.StoreF32x8Part(vv, vel)
		simd.StoreF32x8Part(simd.LoadF32x8Part(w).Add(vv), w)
	}
	archsimd.ClearAVXUpperBits()
}

// verf computes erf(x) per lane with the Abramowitz-Stegun 7.1.26
// polynomial (max absolute error ~1.5e-7, below float32 resolution for this
// use). Symmetry handles negative inputs via sign-bit restore.
func verf(x archsimd.Float32x8) archsimd.Float32x8 {
	signBit := archsimd.BroadcastInt32x8(-0x80000000)
	sign := x.AsInt32x8().And(signBit)
	ax := x.AsInt32x8().AndNot(signBit).AsFloat32x8() // |x|

	one := archsimd.BroadcastFloat32x8(1)
	t := one.Div(archsimd.BroadcastFloat32x8(0.3275911).MulAdd(ax, one))
	p := archsimd.BroadcastFloat32x8(1.061405429)
	p = p.MulAdd(t, archsimd.BroadcastFloat32x8(-1.453152027))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x8(1.421413741))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x8(-0.284496736))
	p = p.MulAdd(t, archsimd.BroadcastFloat32x8(0.254829592))
	p = p.Mul(t)
	zero := archsimd.BroadcastFloat32x8(0)
	e := vexpf(zero.Sub(ax.Mul(ax)))
	erfAbs := one.Sub(p.Mul(e))
	return erfAbs.AsInt32x8().Or(sign).AsFloat32x8()
}

func GeluFwd(dst, src []float32) {
	if !simd.HasAVX2 {
		geluFwdGeneric(dst, src)
		return
	}
	half := archsimd.BroadcastFloat32x8(0.5)
	one := archsimd.BroadcastFloat32x8(1)
	invSqrt2 := archsimd.BroadcastFloat32x8(0.7071067811865476)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return half.Mul(v).Mul(one.Add(verf(v.Mul(invSqrt2))))
	})
}

func GeluBwd(dst, grad, src []float32) {
	if !simd.HasAVX2 {
		geluBwdGeneric(dst, grad, src)
		return
	}
	half := archsimd.BroadcastFloat32x8(0.5)
	one := archsimd.BroadcastFloat32x8(1)
	invSqrt2 := archsimd.BroadcastFloat32x8(0.7071067811865476)
	invSqrt2Pi := archsimd.BroadcastFloat32x8(0.3989422804014327)
	negHalf := archsimd.BroadcastFloat32x8(-0.5)
	mapSlices2(dst, grad, src, func(g, v archsimd.Float32x8) archsimd.Float32x8 {
		cdf := half.Mul(one.Add(verf(v.Mul(invSqrt2))))
		pdf := v.Mul(invSqrt2Pi).Mul(vexpf(negHalf.Mul(v).Mul(v)))
		return g.Mul(cdf.Add(pdf))
	})
}

// hsum reduces a vector accumulator to a scalar.
func hsum(v archsimd.Float32x8) float32 {
	var tmp [8]float32
	simd.StoreF32x8(v, tmp[:])
	return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
}

func LnFwdRow(out, xhat, src, gamma, beta []float32, eps float32) float32 {
	if !simd.HasAVX2 {
		return lnFwdRowGeneric(out, xhat, src, gamma, beta, eps)
	}
	n := float32(len(src))

	var acc archsimd.Float32x8
	s := src
	for len(s) >= 8 {
		acc = acc.Add(simd.LoadF32x8(s))
		s = s[8:]
	}
	if len(s) > 0 {
		v := simd.LoadF32x8Part(s) // missing lanes are zero
		acc = acc.Add(v)
	}
	mean := hsum(acc) / n

	meanV := archsimd.BroadcastFloat32x8(mean)
	var vacc archsimd.Float32x8
	s = src
	for len(s) >= 8 {
		d := simd.LoadF32x8(s).Sub(meanV)
		vacc = d.MulAdd(d, vacc)
		s = s[8:]
	}
	// Scalar tail: zero-filled part-load lanes would skew (v-mean)^2.
	variance := hsum(vacc)
	for _, v := range s {
		d := v - mean
		variance += d * d
	}
	variance /= n
	invStd := 1 / SqrtF(variance+eps)

	invStdV := archsimd.BroadcastFloat32x8(invStd)
	o, xh, sr, ga, be := out, xhat, src, gamma, beta
	for len(sr) >= 8 {
		h := simd.LoadF32x8(sr).Sub(meanV).Mul(invStdV)
		simd.StoreF32x8(h, xh)
		simd.StoreF32x8(h.MulAdd(simd.LoadF32x8(ga), simd.LoadF32x8(be)), o)
		o, xh, sr, ga, be = o[8:], xh[8:], sr[8:], ga[8:], be[8:]
	}
	if len(sr) > 0 {
		sv := simd.LoadF32x8Part(sr)
		gv := simd.LoadF32x8Part(ga)
		bv := simd.LoadF32x8Part(be)
		h := sv.Sub(meanV).Mul(invStdV)
		simd.StoreF32x8Part(h, xh)
		simd.StoreF32x8Part(h.MulAdd(gv, bv), o)
	}
	archsimd.ClearAVXUpperBits()
	return invStd
}

func LnBwdRow(out, g, xhat, gamma, gradGamma, gradBeta []float32, invStd float32) {
	if !simd.HasAVX2 {
		lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta, invStd)
		return
	}
	n := float32(len(g))

	var acc1, acc2 archsimd.Float32x8
	gs, xs, gas, ggs, gbs := g, xhat, gamma, gradGamma, gradBeta
	for len(gs) >= 8 {
		gv := simd.LoadF32x8(gs)
		xh := simd.LoadF32x8(xs)
		simd.StoreF32x8(gv.MulAdd(xh, simd.LoadF32x8(ggs)), ggs)
		simd.StoreF32x8(gv.Add(simd.LoadF32x8(gbs)), gbs)
		dx := gv.Mul(simd.LoadF32x8(gas))
		acc1 = acc1.Add(dx)
		acc2 = dx.MulAdd(xh, acc2)
		gs, xs, gas, ggs, gbs = gs[8:], xs[8:], gas[8:], ggs[8:], gbs[8:]
	}
	if len(gs) > 0 {
		gv := simd.LoadF32x8Part(gs)
		xh := simd.LoadF32x8Part(xs)
		gg := simd.LoadF32x8Part(ggs)
		gb := simd.LoadF32x8Part(gbs)
		gav := simd.LoadF32x8Part(gas)
		simd.StoreF32x8Part(gv.MulAdd(xh, gg), ggs)
		simd.StoreF32x8Part(gv.Add(gb), gbs)
		dx := gv.Mul(gav)
		acc1 = acc1.Add(dx)
		acc2 = dx.MulAdd(xh, acc2)
	}
	sumDXhat := hsum(acc1)
	sumDXhatXhat := hsum(acc2)

	k := invStd / n
	kV := archsimd.BroadcastFloat32x8(k)
	nV := archsimd.BroadcastFloat32x8(n)
	s1V := archsimd.BroadcastFloat32x8(sumDXhat)
	s2V := archsimd.BroadcastFloat32x8(sumDXhatXhat)
	o := out
	gs, xs, gas = g, xhat, gamma
	for len(gs) >= 8 {
		dx := simd.LoadF32x8(gs).Mul(simd.LoadF32x8(gas))
		t := dx.Mul(nV).Sub(s1V).Sub(simd.LoadF32x8(xs).Mul(s2V))
		simd.StoreF32x8(t.Mul(kV), o)
		o, gs, xs, gas = o[8:], gs[8:], xs[8:], gas[8:]
	}
	if len(gs) > 0 {
		gv := simd.LoadF32x8Part(gs)
		xh := simd.LoadF32x8Part(xs)
		gav := simd.LoadF32x8Part(gas)
		dx := gv.Mul(gav)
		simd.StoreF32x8Part(dx.Mul(nV).Sub(s1V).Sub(xh.Mul(s2V)).Mul(kV), o)
	}
	archsimd.ClearAVXUpperBits()
}

// DotVec is the 8-lane FMA dot product; the horizontal sum happens once
// at the end.
func DotVec(a, b []float32) float32 {
	if !simd.HasAVX2 || len(a) < 16 {
		return dotVecGeneric(a, b)
	}
	n := len(a) &^ 7
	var acc archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		acc = simd.LoadF32x8(a[i:]).MulAdd(simd.LoadF32x8(b[i:]), acc)
	}
	var buf [8]float32
	simd.StoreF32x8(acc, buf[:])
	archsimd.ClearAVXUpperBits()
	s := buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
	for i := n; i < len(a); i++ {
		s += a[i] * b[i]
	}
	return s
}

// Axpy computes y += a*x, 8 lanes at a time.
func Axpy(a float32, x, y []float32) {
	if !simd.HasAVX2 || len(x) < 16 {
		axpyGeneric(a, x, y)
		return
	}
	av := archsimd.BroadcastFloat32x8(a)
	n := len(x) &^ 7
	for i := 0; i < n; i += 8 {
		simd.StoreF32x8(simd.LoadF32x8(x[i:]).MulAdd(av, simd.LoadF32x8(y[i:])), y[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < len(x); i++ {
		y[i] += a * x[i]
	}
}

// DotVecs computes out[i] = qs[i*d:(i+1)*d] . k for the len(out) query
// vectors packed contiguously in qs, streaming the shared k once for up
// to four rows per pass. Each row keeps DotVec's exact accumulation
// order — same chunked FMA, same linear horizontal sum, same scalar
// tail — so every result is bit-identical to per-row DotVec.
func DotVecs(qs, k []float32, out []float32) {
	if !simd.HasAVX2 || len(k) < 16 {
		dotVecsGeneric(qs, k, out)
		return
	}
	d := len(k)
	switch len(out) {
	case 8:
		dotVec8(qs, k, out)
		return
	case 7:
		dotVec7(qs, k, out)
		return
	case 6:
		dotVec6(qs, k, out)
		return
	}
	i := 0
	// Eights first: one k load feeds eight dots, and a blocked caller
	// arrives with a multiple of eight.
	for ; i+8 <= len(out); i += 8 {
		dotVec8(qs[i*d:(i+8)*d], k, out[i:i+8])
	}
	for ; i+4 <= len(out); i += 4 {
		dotVec4(qs[i*d:(i+4)*d], k, out[i:i+4])
	}
	if i+2 <= len(out) {
		dotVec2(qs[i*d:(i+2)*d], k, out[i:i+2])
		i += 2
	}
	if i < len(out) {
		out[i] = DotVec(qs[i*d:(i+1)*d], k)
	}
}

func dotVec4(qs, k []float32, out []float32) {
	d := len(k)
	q0 := qs[0*d : 1*d : 1*d]
	q1 := qs[1*d : 2*d : 2*d]
	q2 := qs[2*d : 3*d : 3*d]
	q3 := qs[3*d : 4*d : 4*d]
	n := d &^ 7
	n2 := d &^ 15
	var a0, a1, a2, a3 archsimd.Float32x8
	// Two chunks per iteration: the four loop-carried accumulators make
	// the compiler rotate registers at the back edge, so halving the
	// trips halves that tax; the FMA order per accumulator is unchanged.
	for i := 0; i < n2; i += 16 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		a2 = simd.LoadF32x8(q2[i:]).MulAdd(kv, a2)
		a3 = simd.LoadF32x8(q3[i:]).MulAdd(kv, a3)
		kw := simd.LoadF32x8(k[i+8:])
		a0 = simd.LoadF32x8(q0[i+8:]).MulAdd(kw, a0)
		a1 = simd.LoadF32x8(q1[i+8:]).MulAdd(kw, a1)
		a2 = simd.LoadF32x8(q2[i+8:]).MulAdd(kw, a2)
		a3 = simd.LoadF32x8(q3[i+8:]).MulAdd(kw, a3)
	}
	for i := n2; i < n; i += 8 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		a2 = simd.LoadF32x8(q2[i:]).MulAdd(kv, a2)
		a3 = simd.LoadF32x8(q3[i:]).MulAdd(kv, a3)
	}
	// One buffer per accumulator, all stored before any is read: reusing
	// a single slot chains the horizontal sums through store-to-load
	// dependencies, and with six of them back to back per position there
	// is no surrounding work to hide that latency under.
	var buf0, buf1, buf2, buf3 [8]float32
	simd.StoreF32x8(a0, buf0[:])
	simd.StoreF32x8(a1, buf1[:])
	simd.StoreF32x8(a2, buf2[:])
	simd.StoreF32x8(a3, buf3[:])
	s0 := buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7]
	s1 := buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7]
	s2 := buf2[0] + buf2[1] + buf2[2] + buf2[3] + buf2[4] + buf2[5] + buf2[6] + buf2[7]
	s3 := buf3[0] + buf3[1] + buf3[2] + buf3[3] + buf3[4] + buf3[5] + buf3[6] + buf3[7]
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		s0 += q0[i] * k[i]
		s1 += q1[i] * k[i]
		s2 += q2[i] * k[i]
		s3 += q3[i] * k[i]
	}
	out[0], out[1], out[2], out[3] = s0, s1, s2, s3
}

func dotVec2(qs, k []float32, out []float32) {
	d := len(k)
	q0 := qs[0*d : 1*d : 1*d]
	q1 := qs[1*d : 2*d : 2*d]
	n := d &^ 7
	n2 := d &^ 15
	var a0, a1 archsimd.Float32x8
	for i := 0; i < n2; i += 16 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		kw := simd.LoadF32x8(k[i+8:])
		a0 = simd.LoadF32x8(q0[i+8:]).MulAdd(kw, a0)
		a1 = simd.LoadF32x8(q1[i+8:]).MulAdd(kw, a1)
	}
	for i := n2; i < n; i += 8 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
	}
	var buf0, buf1 [8]float32
	simd.StoreF32x8(a0, buf0[:])
	simd.StoreF32x8(a1, buf1[:])
	s0 := buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7]
	s1 := buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7]
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		s0 += q0[i] * k[i]
		s1 += q1[i] * k[i]
	}
	out[0], out[1] = s0, s1
}

// Axpys accumulates outs[i*d:(i+1)*d] += ws[i] * v for the len(ws) rows
// packed contiguously in outs, streaming the shared v once for up to
// four rows per pass; per row it is bit-identical to Axpy.
func Axpys(ws []float32, v, outs []float32) {
	if !simd.HasAVX2 || len(v) < 16 {
		axpysGeneric(ws, v, outs)
		return
	}
	d := len(v)
	switch len(ws) {
	case 8:
		axpy8(ws, v, outs)
		return
	case 7:
		axpy7(ws, v, outs)
		return
	case 6:
		axpy6(ws, v, outs)
		return
	}
	i := 0
	// Eights first, as in DotVecs: one v load feeds eight rows.
	for ; i+8 <= len(ws); i += 8 {
		axpy8(ws[i:i+8], v, outs[i*d:(i+8)*d])
	}
	for ; i+4 <= len(ws); i += 4 {
		axpy4(ws[i:i+4], v, outs[i*d:(i+4)*d])
	}
	if i+2 <= len(ws) {
		axpy2(ws[i:i+2], v, outs[i*d:(i+2)*d])
		i += 2
	}
	if i < len(ws) {
		Axpy(ws[i], v, outs[i*d:(i+1)*d])
	}
}

func axpy4(ws []float32, v, outs []float32) {
	d := len(v)
	o0 := outs[0*d : 1*d : 1*d]
	o1 := outs[1*d : 2*d : 2*d]
	o2 := outs[2*d : 3*d : 3*d]
	o3 := outs[3*d : 4*d : 4*d]
	w0 := archsimd.BroadcastFloat32x8(ws[0])
	w1 := archsimd.BroadcastFloat32x8(ws[1])
	w2 := archsimd.BroadcastFloat32x8(ws[2])
	w3 := archsimd.BroadcastFloat32x8(ws[3])
	n := d &^ 7
	for i := 0; i < n; i += 8 {
		vv := simd.LoadF32x8(v[i:])
		simd.StoreF32x8(vv.MulAdd(w0, simd.LoadF32x8(o0[i:])), o0[i:])
		simd.StoreF32x8(vv.MulAdd(w1, simd.LoadF32x8(o1[i:])), o1[i:])
		simd.StoreF32x8(vv.MulAdd(w2, simd.LoadF32x8(o2[i:])), o2[i:])
		simd.StoreF32x8(vv.MulAdd(w3, simd.LoadF32x8(o3[i:])), o3[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		o0[i] += ws[0] * v[i]
		o1[i] += ws[1] * v[i]
		o2[i] += ws[2] * v[i]
		o3[i] += ws[3] * v[i]
	}
}

func axpy2(ws []float32, v, outs []float32) {
	d := len(v)
	o0 := outs[0*d : 1*d : 1*d]
	o1 := outs[1*d : 2*d : 2*d]
	w0 := archsimd.BroadcastFloat32x8(ws[0])
	w1 := archsimd.BroadcastFloat32x8(ws[1])
	n := d &^ 7
	for i := 0; i < n; i += 8 {
		vv := simd.LoadF32x8(v[i:])
		simd.StoreF32x8(vv.MulAdd(w0, simd.LoadF32x8(o0[i:])), o0[i:])
		simd.StoreF32x8(vv.MulAdd(w1, simd.LoadF32x8(o1[i:])), o1[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		o0[i] += ws[0] * v[i]
		o1[i] += ws[1] * v[i]
	}
}

// dotVec6/7/8 and axpy6/7/8 are the single-pass forms for the group
// widths real models use (llama 4, qwen 6-7, gpt-oss 8): one call, one
// k stream, one horizontal-sum block per position.
func dotVec6(qs, k []float32, out []float32) {
	d := len(k)
	q0 := qs[0*d : 1*d : 1*d]
	q1 := qs[1*d : 2*d : 2*d]
	q2 := qs[2*d : 3*d : 3*d]
	q3 := qs[3*d : 4*d : 4*d]
	q4 := qs[4*d : 5*d : 5*d]
	q5 := qs[5*d : 6*d : 6*d]
	n := d &^ 7
	var a0, a1, a2, a3, a4, a5 archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		a2 = simd.LoadF32x8(q2[i:]).MulAdd(kv, a2)
		a3 = simd.LoadF32x8(q3[i:]).MulAdd(kv, a3)
		a4 = simd.LoadF32x8(q4[i:]).MulAdd(kv, a4)
		a5 = simd.LoadF32x8(q5[i:]).MulAdd(kv, a5)
	}
	var buf0, buf1, buf2, buf3, buf4, buf5 [8]float32
	simd.StoreF32x8(a0, buf0[:])
	simd.StoreF32x8(a1, buf1[:])
	simd.StoreF32x8(a2, buf2[:])
	simd.StoreF32x8(a3, buf3[:])
	simd.StoreF32x8(a4, buf4[:])
	simd.StoreF32x8(a5, buf5[:])
	s0 := buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7]
	s1 := buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7]
	s2 := buf2[0] + buf2[1] + buf2[2] + buf2[3] + buf2[4] + buf2[5] + buf2[6] + buf2[7]
	s3 := buf3[0] + buf3[1] + buf3[2] + buf3[3] + buf3[4] + buf3[5] + buf3[6] + buf3[7]
	s4 := buf4[0] + buf4[1] + buf4[2] + buf4[3] + buf4[4] + buf4[5] + buf4[6] + buf4[7]
	s5 := buf5[0] + buf5[1] + buf5[2] + buf5[3] + buf5[4] + buf5[5] + buf5[6] + buf5[7]
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		s0 += q0[i] * k[i]
		s1 += q1[i] * k[i]
		s2 += q2[i] * k[i]
		s3 += q3[i] * k[i]
		s4 += q4[i] * k[i]
		s5 += q5[i] * k[i]
	}
	out[0] = s0
	out[1] = s1
	out[2] = s2
	out[3] = s3
	out[4] = s4
	out[5] = s5
}

func dotVec7(qs, k []float32, out []float32) {
	d := len(k)
	q0 := qs[0*d : 1*d : 1*d]
	q1 := qs[1*d : 2*d : 2*d]
	q2 := qs[2*d : 3*d : 3*d]
	q3 := qs[3*d : 4*d : 4*d]
	q4 := qs[4*d : 5*d : 5*d]
	q5 := qs[5*d : 6*d : 6*d]
	q6 := qs[6*d : 7*d : 7*d]
	n := d &^ 7
	var a0, a1, a2, a3, a4, a5, a6 archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		a2 = simd.LoadF32x8(q2[i:]).MulAdd(kv, a2)
		a3 = simd.LoadF32x8(q3[i:]).MulAdd(kv, a3)
		a4 = simd.LoadF32x8(q4[i:]).MulAdd(kv, a4)
		a5 = simd.LoadF32x8(q5[i:]).MulAdd(kv, a5)
		a6 = simd.LoadF32x8(q6[i:]).MulAdd(kv, a6)
	}
	var buf0, buf1, buf2, buf3, buf4, buf5, buf6 [8]float32
	simd.StoreF32x8(a0, buf0[:])
	simd.StoreF32x8(a1, buf1[:])
	simd.StoreF32x8(a2, buf2[:])
	simd.StoreF32x8(a3, buf3[:])
	simd.StoreF32x8(a4, buf4[:])
	simd.StoreF32x8(a5, buf5[:])
	simd.StoreF32x8(a6, buf6[:])
	s0 := buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7]
	s1 := buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7]
	s2 := buf2[0] + buf2[1] + buf2[2] + buf2[3] + buf2[4] + buf2[5] + buf2[6] + buf2[7]
	s3 := buf3[0] + buf3[1] + buf3[2] + buf3[3] + buf3[4] + buf3[5] + buf3[6] + buf3[7]
	s4 := buf4[0] + buf4[1] + buf4[2] + buf4[3] + buf4[4] + buf4[5] + buf4[6] + buf4[7]
	s5 := buf5[0] + buf5[1] + buf5[2] + buf5[3] + buf5[4] + buf5[5] + buf5[6] + buf5[7]
	s6 := buf6[0] + buf6[1] + buf6[2] + buf6[3] + buf6[4] + buf6[5] + buf6[6] + buf6[7]
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		s0 += q0[i] * k[i]
		s1 += q1[i] * k[i]
		s2 += q2[i] * k[i]
		s3 += q3[i] * k[i]
		s4 += q4[i] * k[i]
		s5 += q5[i] * k[i]
		s6 += q6[i] * k[i]
	}
	out[0] = s0
	out[1] = s1
	out[2] = s2
	out[3] = s3
	out[4] = s4
	out[5] = s5
	out[6] = s6
}

func dotVec8(qs, k []float32, out []float32) {
	d := len(k)
	q0 := qs[0*d : 1*d : 1*d]
	q1 := qs[1*d : 2*d : 2*d]
	q2 := qs[2*d : 3*d : 3*d]
	q3 := qs[3*d : 4*d : 4*d]
	q4 := qs[4*d : 5*d : 5*d]
	q5 := qs[5*d : 6*d : 6*d]
	q6 := qs[6*d : 7*d : 7*d]
	q7 := qs[7*d : 8*d : 8*d]
	n := d &^ 7
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		kv := simd.LoadF32x8(k[i:])
		a0 = simd.LoadF32x8(q0[i:]).MulAdd(kv, a0)
		a1 = simd.LoadF32x8(q1[i:]).MulAdd(kv, a1)
		a2 = simd.LoadF32x8(q2[i:]).MulAdd(kv, a2)
		a3 = simd.LoadF32x8(q3[i:]).MulAdd(kv, a3)
		a4 = simd.LoadF32x8(q4[i:]).MulAdd(kv, a4)
		a5 = simd.LoadF32x8(q5[i:]).MulAdd(kv, a5)
		a6 = simd.LoadF32x8(q6[i:]).MulAdd(kv, a6)
		a7 = simd.LoadF32x8(q7[i:]).MulAdd(kv, a7)
	}
	var buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7 [8]float32
	simd.StoreF32x8(a0, buf0[:])
	simd.StoreF32x8(a1, buf1[:])
	simd.StoreF32x8(a2, buf2[:])
	simd.StoreF32x8(a3, buf3[:])
	simd.StoreF32x8(a4, buf4[:])
	simd.StoreF32x8(a5, buf5[:])
	simd.StoreF32x8(a6, buf6[:])
	simd.StoreF32x8(a7, buf7[:])
	s0 := buf0[0] + buf0[1] + buf0[2] + buf0[3] + buf0[4] + buf0[5] + buf0[6] + buf0[7]
	s1 := buf1[0] + buf1[1] + buf1[2] + buf1[3] + buf1[4] + buf1[5] + buf1[6] + buf1[7]
	s2 := buf2[0] + buf2[1] + buf2[2] + buf2[3] + buf2[4] + buf2[5] + buf2[6] + buf2[7]
	s3 := buf3[0] + buf3[1] + buf3[2] + buf3[3] + buf3[4] + buf3[5] + buf3[6] + buf3[7]
	s4 := buf4[0] + buf4[1] + buf4[2] + buf4[3] + buf4[4] + buf4[5] + buf4[6] + buf4[7]
	s5 := buf5[0] + buf5[1] + buf5[2] + buf5[3] + buf5[4] + buf5[5] + buf5[6] + buf5[7]
	s6 := buf6[0] + buf6[1] + buf6[2] + buf6[3] + buf6[4] + buf6[5] + buf6[6] + buf6[7]
	s7 := buf7[0] + buf7[1] + buf7[2] + buf7[3] + buf7[4] + buf7[5] + buf7[6] + buf7[7]
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		s0 += q0[i] * k[i]
		s1 += q1[i] * k[i]
		s2 += q2[i] * k[i]
		s3 += q3[i] * k[i]
		s4 += q4[i] * k[i]
		s5 += q5[i] * k[i]
		s6 += q6[i] * k[i]
		s7 += q7[i] * k[i]
	}
	out[0] = s0
	out[1] = s1
	out[2] = s2
	out[3] = s3
	out[4] = s4
	out[5] = s5
	out[6] = s6
	out[7] = s7
}

func axpy6(ws []float32, v, outs []float32) {
	d := len(v)
	o0 := outs[0*d : 1*d : 1*d]
	o1 := outs[1*d : 2*d : 2*d]
	o2 := outs[2*d : 3*d : 3*d]
	o3 := outs[3*d : 4*d : 4*d]
	o4 := outs[4*d : 5*d : 5*d]
	o5 := outs[5*d : 6*d : 6*d]
	w0 := archsimd.BroadcastFloat32x8(ws[0])
	w1 := archsimd.BroadcastFloat32x8(ws[1])
	w2 := archsimd.BroadcastFloat32x8(ws[2])
	w3 := archsimd.BroadcastFloat32x8(ws[3])
	w4 := archsimd.BroadcastFloat32x8(ws[4])
	w5 := archsimd.BroadcastFloat32x8(ws[5])
	n := d &^ 7
	for i := 0; i < n; i += 8 {
		vv := simd.LoadF32x8(v[i:])
		simd.StoreF32x8(vv.MulAdd(w0, simd.LoadF32x8(o0[i:])), o0[i:])
		simd.StoreF32x8(vv.MulAdd(w1, simd.LoadF32x8(o1[i:])), o1[i:])
		simd.StoreF32x8(vv.MulAdd(w2, simd.LoadF32x8(o2[i:])), o2[i:])
		simd.StoreF32x8(vv.MulAdd(w3, simd.LoadF32x8(o3[i:])), o3[i:])
		simd.StoreF32x8(vv.MulAdd(w4, simd.LoadF32x8(o4[i:])), o4[i:])
		simd.StoreF32x8(vv.MulAdd(w5, simd.LoadF32x8(o5[i:])), o5[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		o0[i] += ws[0] * v[i]
		o1[i] += ws[1] * v[i]
		o2[i] += ws[2] * v[i]
		o3[i] += ws[3] * v[i]
		o4[i] += ws[4] * v[i]
		o5[i] += ws[5] * v[i]
	}
}

func axpy7(ws []float32, v, outs []float32) {
	d := len(v)
	o0 := outs[0*d : 1*d : 1*d]
	o1 := outs[1*d : 2*d : 2*d]
	o2 := outs[2*d : 3*d : 3*d]
	o3 := outs[3*d : 4*d : 4*d]
	o4 := outs[4*d : 5*d : 5*d]
	o5 := outs[5*d : 6*d : 6*d]
	o6 := outs[6*d : 7*d : 7*d]
	w0 := archsimd.BroadcastFloat32x8(ws[0])
	w1 := archsimd.BroadcastFloat32x8(ws[1])
	w2 := archsimd.BroadcastFloat32x8(ws[2])
	w3 := archsimd.BroadcastFloat32x8(ws[3])
	w4 := archsimd.BroadcastFloat32x8(ws[4])
	w5 := archsimd.BroadcastFloat32x8(ws[5])
	w6 := archsimd.BroadcastFloat32x8(ws[6])
	n := d &^ 7
	for i := 0; i < n; i += 8 {
		vv := simd.LoadF32x8(v[i:])
		simd.StoreF32x8(vv.MulAdd(w0, simd.LoadF32x8(o0[i:])), o0[i:])
		simd.StoreF32x8(vv.MulAdd(w1, simd.LoadF32x8(o1[i:])), o1[i:])
		simd.StoreF32x8(vv.MulAdd(w2, simd.LoadF32x8(o2[i:])), o2[i:])
		simd.StoreF32x8(vv.MulAdd(w3, simd.LoadF32x8(o3[i:])), o3[i:])
		simd.StoreF32x8(vv.MulAdd(w4, simd.LoadF32x8(o4[i:])), o4[i:])
		simd.StoreF32x8(vv.MulAdd(w5, simd.LoadF32x8(o5[i:])), o5[i:])
		simd.StoreF32x8(vv.MulAdd(w6, simd.LoadF32x8(o6[i:])), o6[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		o0[i] += ws[0] * v[i]
		o1[i] += ws[1] * v[i]
		o2[i] += ws[2] * v[i]
		o3[i] += ws[3] * v[i]
		o4[i] += ws[4] * v[i]
		o5[i] += ws[5] * v[i]
		o6[i] += ws[6] * v[i]
	}
}

func axpy8(ws []float32, v, outs []float32) {
	d := len(v)
	o0 := outs[0*d : 1*d : 1*d]
	o1 := outs[1*d : 2*d : 2*d]
	o2 := outs[2*d : 3*d : 3*d]
	o3 := outs[3*d : 4*d : 4*d]
	o4 := outs[4*d : 5*d : 5*d]
	o5 := outs[5*d : 6*d : 6*d]
	o6 := outs[6*d : 7*d : 7*d]
	o7 := outs[7*d : 8*d : 8*d]
	w0 := archsimd.BroadcastFloat32x8(ws[0])
	w1 := archsimd.BroadcastFloat32x8(ws[1])
	w2 := archsimd.BroadcastFloat32x8(ws[2])
	w3 := archsimd.BroadcastFloat32x8(ws[3])
	w4 := archsimd.BroadcastFloat32x8(ws[4])
	w5 := archsimd.BroadcastFloat32x8(ws[5])
	w6 := archsimd.BroadcastFloat32x8(ws[6])
	w7 := archsimd.BroadcastFloat32x8(ws[7])
	n := d &^ 7
	for i := 0; i < n; i += 8 {
		vv := simd.LoadF32x8(v[i:])
		simd.StoreF32x8(vv.MulAdd(w0, simd.LoadF32x8(o0[i:])), o0[i:])
		simd.StoreF32x8(vv.MulAdd(w1, simd.LoadF32x8(o1[i:])), o1[i:])
		simd.StoreF32x8(vv.MulAdd(w2, simd.LoadF32x8(o2[i:])), o2[i:])
		simd.StoreF32x8(vv.MulAdd(w3, simd.LoadF32x8(o3[i:])), o3[i:])
		simd.StoreF32x8(vv.MulAdd(w4, simd.LoadF32x8(o4[i:])), o4[i:])
		simd.StoreF32x8(vv.MulAdd(w5, simd.LoadF32x8(o5[i:])), o5[i:])
		simd.StoreF32x8(vv.MulAdd(w6, simd.LoadF32x8(o6[i:])), o6[i:])
		simd.StoreF32x8(vv.MulAdd(w7, simd.LoadF32x8(o7[i:])), o7[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < d; i++ {
		o0[i] += ws[0] * v[i]
		o1[i] += ws[1] * v[i]
		o2[i] += ws[2] * v[i]
		o3[i] += ws[3] * v[i]
		o4[i] += ws[4] * v[i]
		o5[i] += ws[5] * v[i]
		o6[i] += ws[6] * v[i]
		o7[i] += ws[7] * v[i]
	}
}
