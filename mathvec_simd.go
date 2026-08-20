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
