//go:build goexperiment.simd && amd64

package tensai

import "simd/archsimd"

// AVX2 versions of the element-wise kernels. Bodies stay free of scalar
// float instructions (SSE encodings) while the vector upper state is dirty
// — see dot_simd.go — and every kernel clears that state before returning.

// vexpf computes e^x per lane with a Cephes-style degree-5 polynomial after
// range reduction x = z*ln2 + r. Inputs are clamped to about [-87, 88], so
// the result saturates instead of overflowing to Inf; relative error is a
// few float32 ulps, far below what training precision requires.
func vexpf(x archsimd.Float32x8) archsimd.Float32x8 {
	x = x.Min(archsimd.BroadcastFloat32x8(88.0)).Max(archsimd.BroadcastFloat32x8(-87.0))
	z := roundEven(x.Mul(archsimd.BroadcastFloat32x8(1.44269504)))
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
func mapSlices(dst, src []Float, f func(archsimd.Float32x8) archsimd.Float32x8) {
	for len(dst) >= 8 && len(src) >= 8 {
		storeF32x8(f(loadF32x8(src)), dst)
		dst, src = dst[8:], src[8:]
	}
	if len(dst) > 0 {
		storeF32x8Part(f(loadF32x8Part(src)), dst)
	}
	archsimd.ClearAVXUpperBits()
}

// mapSlices2 is mapSlices over two inputs.
func mapSlices2(dst, x, y []Float, f func(a, b archsimd.Float32x8) archsimd.Float32x8) {
	for len(dst) >= 8 && len(x) >= 8 && len(y) >= 8 {
		storeF32x8(f(loadF32x8(x), loadF32x8(y)), dst)
		dst, x, y = dst[8:], x[8:], y[8:]
	}
	if len(dst) > 0 {
		storeF32x8Part(f(loadF32x8Part(x), loadF32x8Part(y)), dst)
	}
	archsimd.ClearAVXUpperBits()
}

func reluFwd(dst, src []Float) {
	if !hasAVX2 {
		reluFwdGeneric(dst, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Max(zero)
	})
}

func reluBwd(dst, grad, src []Float) {
	if !hasAVX2 {
		reluBwdGeneric(dst, grad, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	mapSlices2(dst, grad, src, func(g, v archsimd.Float32x8) archsimd.Float32x8 {
		return g.AsInt32x8().And(v.Greater(zero).ToInt32x8()).AsFloat32x8()
	})
}

func leakyFwd(dst, src []Float, alpha Float) {
	if !hasAVX2 {
		leakyFwdGeneric(dst, src, alpha)
		return
	}
	av := archsimd.BroadcastFloat32x8(alpha)
	// max(x, alpha*x) equals leaky ReLU for alpha in [0, 1).
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Max(v.Mul(av))
	})
}

func leakyBwd(dst, grad, src []Float, alpha Float) {
	if !hasAVX2 {
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

func sigmoidFwd(dst, src []Float) {
	if !hasAVX2 {
		sigmoidFwdGeneric(dst, src)
		return
	}
	zero := archsimd.BroadcastFloat32x8(0)
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return one.Div(one.Add(vexpf(zero.Sub(v))))
	})
}

func sigmoidBwd(dst, grad, y []Float) {
	if !hasAVX2 {
		sigmoidBwdGeneric(dst, grad, y)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices2(dst, grad, y, func(g, yv archsimd.Float32x8) archsimd.Float32x8 {
		return g.Mul(yv).Mul(one.Sub(yv))
	})
}

func tanhFwd(dst, src []Float) {
	if !hasAVX2 {
		tanhFwdGeneric(dst, src)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		e2 := vexpf(v.Add(v))
		return e2.Sub(one).Div(e2.Add(one))
	})
}

func tanhBwd(dst, grad, y []Float) {
	if !hasAVX2 {
		tanhBwdGeneric(dst, grad, y)
		return
	}
	one := archsimd.BroadcastFloat32x8(1)
	mapSlices2(dst, grad, y, func(g, yv archsimd.Float32x8) archsimd.Float32x8 {
		return g.Mul(one.Sub(yv.Mul(yv)))
	})
}

func expShift(dst, src []Float, shift Float) {
	if !hasAVX2 {
		expShiftGeneric(dst, src, shift)
		return
	}
	sv := archsimd.BroadcastFloat32x8(shift)
	mapSlices(dst, src, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return vexpf(v.Sub(sv))
	})
}

func addSlice(dst, src []Float) {
	if !hasAVX2 {
		addSliceGeneric(dst, src)
		return
	}
	mapSlices2(dst, dst, src, func(d, s archsimd.Float32x8) archsimd.Float32x8 {
		return d.Add(s)
	})
}

func scaleSlice(dst []Float, s Float) {
	if !hasAVX2 {
		scaleSliceGeneric(dst, s)
		return
	}
	sv := archsimd.BroadcastFloat32x8(s)
	mapSlices(dst, dst, func(v archsimd.Float32x8) archsimd.Float32x8 {
		return v.Mul(sv)
	})
}

func adamStepSlice(w, g, m, v []Float, beta1, beta2, rc1, rc2, lr, eps, wd Float) {
	if !hasAVX2 {
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
		wv, mv, vv := step(loadF32x8(w), loadF32x8(g), loadF32x8(m), loadF32x8(v))
		storeF32x8(wv, w)
		storeF32x8(mv, m)
		storeF32x8(vv, v)
		w, g, m, v = w[8:], g[8:], m[8:], v[8:]
	}
	if len(w) > 0 {
		wv, mv, vv := step(loadF32x8Part(w), loadF32x8Part(g), loadF32x8Part(m), loadF32x8Part(v))
		storeF32x8Part(wv, w)
		storeF32x8Part(mv, m)
		storeF32x8Part(vv, v)
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

func geluFwd(dst, src []Float) {
	if !hasAVX2 {
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

func geluBwd(dst, grad, src []Float) {
	if !hasAVX2 {
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
func hsum(v archsimd.Float32x8) Float {
	var tmp [8]Float
	storeF32x8(v, tmp[:])
	return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
}

func lnFwdRow(out, xhat, src, gamma, beta []Float, eps Float) Float {
	if !hasAVX2 {
		return lnFwdRowGeneric(out, xhat, src, gamma, beta, eps)
	}
	n := Float(len(src))

	var acc archsimd.Float32x8
	s := src
	for len(s) >= 8 {
		acc = acc.Add(loadF32x8(s))
		s = s[8:]
	}
	if len(s) > 0 {
		v := loadF32x8Part(s) // missing lanes are zero
		acc = acc.Add(v)
	}
	mean := hsum(acc) / n

	meanV := archsimd.BroadcastFloat32x8(mean)
	var vacc archsimd.Float32x8
	s = src
	for len(s) >= 8 {
		d := loadF32x8(s).Sub(meanV)
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
	invStd := 1 / sqrtF(variance+eps)

	invStdV := archsimd.BroadcastFloat32x8(invStd)
	o, xh, sr, ga, be := out, xhat, src, gamma, beta
	for len(sr) >= 8 {
		h := loadF32x8(sr).Sub(meanV).Mul(invStdV)
		storeF32x8(h, xh)
		storeF32x8(h.MulAdd(loadF32x8(ga), loadF32x8(be)), o)
		o, xh, sr, ga, be = o[8:], xh[8:], sr[8:], ga[8:], be[8:]
	}
	if len(sr) > 0 {
		sv := loadF32x8Part(sr)
		gv := loadF32x8Part(ga)
		bv := loadF32x8Part(be)
		h := sv.Sub(meanV).Mul(invStdV)
		storeF32x8Part(h, xh)
		storeF32x8Part(h.MulAdd(gv, bv), o)
	}
	archsimd.ClearAVXUpperBits()
	return invStd
}

func lnBwdRow(out, g, xhat, gamma, gradGamma, gradBeta []Float, invStd Float) {
	if !hasAVX2 {
		lnBwdRowGeneric(out, g, xhat, gamma, gradGamma, gradBeta, invStd)
		return
	}
	n := Float(len(g))

	var acc1, acc2 archsimd.Float32x8
	gs, xs, gas, ggs, gbs := g, xhat, gamma, gradGamma, gradBeta
	for len(gs) >= 8 {
		gv := loadF32x8(gs)
		xh := loadF32x8(xs)
		storeF32x8(gv.MulAdd(xh, loadF32x8(ggs)), ggs)
		storeF32x8(gv.Add(loadF32x8(gbs)), gbs)
		dx := gv.Mul(loadF32x8(gas))
		acc1 = acc1.Add(dx)
		acc2 = dx.MulAdd(xh, acc2)
		gs, xs, gas, ggs, gbs = gs[8:], xs[8:], gas[8:], ggs[8:], gbs[8:]
	}
	if len(gs) > 0 {
		gv := loadF32x8Part(gs)
		xh := loadF32x8Part(xs)
		gg := loadF32x8Part(ggs)
		gb := loadF32x8Part(gbs)
		gav := loadF32x8Part(gas)
		storeF32x8Part(gv.MulAdd(xh, gg), ggs)
		storeF32x8Part(gv.Add(gb), gbs)
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
		dx := loadF32x8(gs).Mul(loadF32x8(gas))
		t := dx.Mul(nV).Sub(s1V).Sub(loadF32x8(xs).Mul(s2V))
		storeF32x8(t.Mul(kV), o)
		o, gs, xs, gas = o[8:], gs[8:], xs[8:], gas[8:]
	}
	if len(gs) > 0 {
		gv := loadF32x8Part(gs)
		xh := loadF32x8Part(xs)
		gav := loadF32x8Part(gas)
		dx := gv.Mul(gav)
		storeF32x8Part(dx.Mul(nV).Sub(s1V).Sub(xh.Mul(s2V)).Mul(kV), o)
	}
	archsimd.ClearAVXUpperBits()
}

// dotVec is the 8-lane FMA dot product; the horizontal sum happens once
// at the end.
func dotVec(a, b []Float) Float {
	if !hasAVX2 || len(a) < 16 {
		return dotVecGeneric(a, b)
	}
	n := len(a) &^ 7
	var acc archsimd.Float32x8
	for i := 0; i < n; i += 8 {
		acc = loadF32x8(a[i:]).MulAdd(loadF32x8(b[i:]), acc)
	}
	var buf [8]Float
	storeF32x8(acc, buf[:])
	archsimd.ClearAVXUpperBits()
	s := buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
	for i := n; i < len(a); i++ {
		s += a[i] * b[i]
	}
	return s
}

// axpy computes y += a*x, 8 lanes at a time.
func axpy(a Float, x, y []Float) {
	if !hasAVX2 || len(x) < 16 {
		axpyGeneric(a, x, y)
		return
	}
	av := archsimd.BroadcastFloat32x8(a)
	n := len(x) &^ 7
	for i := 0; i < n; i += 8 {
		storeF32x8(loadF32x8(x[i:]).MulAdd(av, loadF32x8(y[i:])), y[i:])
	}
	archsimd.ClearAVXUpperBits()
	for i := n; i < len(x); i++ {
		y[i] += a * x[i]
	}
}
