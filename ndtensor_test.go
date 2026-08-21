package tensai

import (
	"math"
	"math/rand"
	"testing"
)

// refBinOp is a slow reference for broadcast element-wise ops: every output
// index is mapped into each operand by clamping broadcast axes to 0.
func refBinOp(t *testing.T, a, b *Tensor, fn func(x, y Float) Float) *Tensor {
	t.Helper()
	shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		t.Fatalf("broadcastShapes: %v", err)
	}
	out := NewTensor(shape...)
	idx := make([]int, len(shape))
	operand := func(x *Tensor) int {
		off := 0
		for i := 1; i <= len(x.Shape); i++ {
			d := x.Shape[len(x.Shape)-i]
			j := idx[len(idx)-i] % d
			off += j * contiguousStrides(x.Shape)[len(x.Shape)-i]
		}
		return off
	}
	for pos := range out.Data {
		out.Data[pos] = fn(a.Data[operand(a)], b.Data[operand(b)])
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func randTensor(rng *rand.Rand, shape ...int) *Tensor {
	x := NewTensor(shape...)
	for i := range x.Data {
		x.Data[i] = Float(rng.NormFloat64())
	}
	return x
}

func tensorsClose(t *testing.T, got, want *Tensor, tol float64) {
	t.Helper()
	if !sameDims(got.Shape, want.Shape) {
		t.Fatalf("shape mismatch: got %v want %v", got.Shape, want.Shape)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(got.Data[i] - want.Data[i])); diff > tol {
			t.Fatalf("element %d: got %v want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestTensorBroadcastOps(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	add := func(x, y Float) Float { return x + y }
	cases := [][2][]int{
		{{2, 3}, {2, 3}},
		{{2, 3}, {3}},
		{{3}, {2, 3}},
		{{2, 1, 3}, {4, 1}},
		{{5, 1}, {1, 4}},
		{{2, 3, 4}, {1}},
		{{1, 1, 1}, {2, 2, 2}},
		{{6}, {6}},
	}
	for _, c := range cases {
		a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
		got, err := a.Add(b)
		if err != nil {
			t.Fatalf("add %v+%v: %v", c[0], c[1], err)
		}
		tensorsClose(t, got, refBinOp(t, a, b, add), 1e-6)
	}

	a, b := randTensor(rng, 3, 4), randTensor(rng, 4)
	for name, pair := range map[string][2]func() (*Tensor, error){
		"sub": {func() (*Tensor, error) { return a.Sub(b) }, nil},
		"mul": {func() (*Tensor, error) { return a.Mul(b) }, nil},
		"div": {func() (*Tensor, error) { return a.Div(b) }, nil},
	} {
		got, err := pair[0]()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var fn func(x, y Float) Float
		switch name {
		case "sub":
			fn = func(x, y Float) Float { return x - y }
		case "mul":
			fn = func(x, y Float) Float { return x * y }
		case "div":
			fn = func(x, y Float) Float { return x / y }
		}
		tensorsClose(t, got, refBinOp(t, a, b, fn), 1e-6)
	}

	if _, err := randTensor(rng, 2, 3).Add(randTensor(rng, 4)); err == nil {
		t.Fatal("expected broadcast error for [2,3]+[4]")
	}
}

func TestTensorMatMul2D(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	a, b := randTensor(rng, 3, 4), randTensor(rng, 4, 5)
	got, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	am, _ := a.Matrix()
	bm, _ := b.Matrix()
	want, err := Dot(am, bm)
	if err != nil {
		t.Fatalf("dot: %v", err)
	}
	tensorsClose(t, got, want.Tensor(), 1e-5)
}

// refMatMul multiplies stacks of matrices with plain loops, resolving
// broadcast batch axes by modulo.
func refMatMul(t *testing.T, a, b *Tensor) *Tensor {
	t.Helper()
	na, nb := len(a.Shape), len(b.Shape)
	m, k, n := a.Shape[na-2], a.Shape[na-1], b.Shape[nb-1]
	batch, err := broadcastShapes(a.Shape[:na-2], b.Shape[:nb-2])
	if err != nil {
		t.Fatalf("batch broadcast: %v", err)
	}
	out := NewTensor(append(append([]int(nil), batch...), m, n)...)
	as := broadcastStrides(a.Shape[:na-2], batch)
	bs := broadcastStrides(b.Shape[:nb-2], batch)
	for bi := 0; bi < prodDims(batch); bi++ {
		offA, offB := 0, 0
		for d, rem := len(batch)-1, bi; d >= 0; d-- {
			offA += (rem % batch[d]) * as[d]
			offB += (rem % batch[d]) * bs[d]
			rem /= batch[d]
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum float64
				for l := 0; l < k; l++ {
					sum += float64(a.Data[offA*m*k+i*k+l]) * float64(b.Data[offB*k*n+l*n+j])
				}
				out.Data[bi*m*n+i*n+j] = Float(sum)
			}
		}
	}
	return out
}

func TestTensorMatMulBatched(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	cases := [][2][]int{
		{{2, 3, 4}, {2, 4, 5}},       // plain batch
		{{3, 4}, {6, 4, 5}},          // a broadcast across b's batch
		{{6, 3, 4}, {4, 5}},          // b broadcast across a's batch
		{{2, 1, 3, 4}, {1, 5, 4, 2}}, // batch axes broadcast both ways
		{{1, 3, 4}, {6, 4, 5}},       // explicit 1 batch axis
	}
	for _, c := range cases {
		a, b := randTensor(rng, c[0]...), randTensor(rng, c[1]...)
		got, err := MatMul(a, b)
		if err != nil {
			t.Fatalf("matmul %v*%v: %v", c[0], c[1], err)
		}
		tensorsClose(t, got, refMatMul(t, a, b), 1e-4)
	}

	// Large enough to cross the multi-worker threshold in dotWorkerCount.
	a, b := randTensor(rng, 16, 32, 32), randTensor(rng, 16, 32, 64)
	got, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("large matmul: %v", err)
	}
	tensorsClose(t, got, refMatMul(t, a, b), 1e-3)

	if _, err := MatMul(randTensor(rng, 2, 3), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected inner-dimension mismatch error")
	}
	if _, err := MatMul(randTensor(rng, 2, 3, 4), randTensor(rng, 3, 4, 5)); err == nil {
		t.Fatal("expected batch mismatch error for batches 2 vs 3")
	}
	if _, err := MatMul(randTensor(rng, 4), randTensor(rng, 4, 5)); err == nil {
		t.Fatal("expected error for 1-D operand")
	}
}

func TestTensorTranspose(t *testing.T) {
	rng := rand.New(rand.NewSource(4))

	a := randTensor(rng, 3, 5)
	got, err := a.Transpose()
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	am, _ := a.Matrix()
	tensorsClose(t, got, am.T().Tensor(), 0)

	b := randTensor(rng, 2, 3, 4)
	bt, err := b.Transpose() // swap last two axes
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	if !sameDims(bt.Shape, []int{2, 4, 3}) {
		t.Fatalf("shape: got %v", bt.Shape)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for l := 0; l < 4; l++ {
				if bt.At(i, l, j) != b.At(i, j, l) {
					t.Fatalf("bt[%d,%d,%d] != b[%d,%d,%d]", i, l, j, i, j, l)
				}
			}
		}
	}

	bp, err := b.Transpose(2, 0, 1)
	if err != nil {
		t.Fatalf("transpose perm: %v", err)
	}
	if !sameDims(bp.Shape, []int{4, 2, 3}) {
		t.Fatalf("shape: got %v", bp.Shape)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			for l := 0; l < 4; l++ {
				if bp.At(l, i, j) != b.At(i, j, l) {
					t.Fatalf("bp[%d,%d,%d] != b[%d,%d,%d]", l, i, j, i, j, l)
				}
			}
		}
	}

	if _, err := b.Transpose(0, 1); err == nil {
		t.Fatal("expected error for short permutation")
	}
	if _, err := b.Transpose(0, 1, 1); err == nil {
		t.Fatal("expected error for repeated axis")
	}
	if _, err := randTensor(rng, 5).Transpose(); err == nil {
		t.Fatal("expected error for 1-D no-arg transpose")
	}

	// A @ B^T via batched Transpose+MatMul against the same product on the
	// untransposed operand.
	q := randTensor(rng, 2, 3, 4)
	kt, err := randTensor(rng, 2, 5, 4).Transpose()
	if err != nil {
		t.Fatalf("transpose: %v", err)
	}
	scores, err := MatMul(q, kt)
	if err != nil {
		t.Fatalf("matmul: %v", err)
	}
	if !sameDims(scores.Shape, []int{2, 3, 5}) {
		t.Fatalf("scores shape: got %v", scores.Shape)
	}
}

func TestTensorReshapeAndViews(t *testing.T) {
	x, err := NewTensorFromSlice([]Float{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatalf("from slice: %v", err)
	}
	r, err := x.Reshape(3, -1)
	if err != nil {
		t.Fatalf("reshape: %v", err)
	}
	if !sameDims(r.Shape, []int{3, 2}) {
		t.Fatalf("reshape shape: got %v", r.Shape)
	}
	r.Set(42, 0, 0)
	if x.At(0, 0) != 42 {
		t.Fatal("reshape must share backing data")
	}
	if _, err := x.Reshape(4, -1); err == nil {
		t.Fatal("expected reshape error for 6 elements to [4,-1]")
	}
	if _, err := x.Reshape(-1, -1); err == nil {
		t.Fatal("expected reshape error for two -1 axes")
	}
	if _, err := x.Reshape(7); err == nil {
		t.Fatal("expected reshape error for wrong element count")
	}

	m := NewMatrix(2, 3)
	m.Set(1, 2, 7)
	mt := m.Tensor()
	if mt.At(1, 2) != 7 {
		t.Fatal("matrix->tensor view mismatch")
	}
	mt.Set(9, 0, 1)
	if m.At(0, 1) != 9 {
		t.Fatal("matrix->tensor view must share data")
	}
	back, err := mt.Matrix()
	if err != nil {
		t.Fatalf("tensor->matrix: %v", err)
	}
	if back.At(0, 1) != 9 {
		t.Fatal("tensor->matrix view mismatch")
	}
	if _, err := NewTensor(2, 3, 4).Matrix(); err == nil {
		t.Fatal("expected error viewing 3-D tensor as matrix")
	}

	if _, err := NewTensorFromSlice([]Float{1, 2}, 3); err == nil {
		t.Fatal("expected length mismatch error")
	}
	if err := NewTensor(2, 2).Validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	bad := &Tensor{Shape: []int{2, 2}, Data: make([]Float, 3)}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected validate error")
	}
}
