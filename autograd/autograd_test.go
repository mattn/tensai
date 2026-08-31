package autograd

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/optim"
)

// checkParamGrad compares the analytic gradient of loss w.r.t. param against
// finite differences. build must construct the graph fresh from the current
// matrix contents and return (loss node, param node).
func checkParamGrad(t *testing.T, param *tensai.Tensor, build func() (*Node, *Node), name string) {
	t.Helper()
	loss, p := build()
	// Param nodes may be shared across successive checks; Backward
	// accumulates, so start from a clean gradient.
	ZeroGrads(p)
	loss.Backward()
	if p.Grad == nil {
		t.Fatalf("%s: no gradient computed", name)
	}
	analytic := make([]float64, len(p.Grad.Data))
	for i, g := range p.Grad.Data {
		analytic[i] = float64(g)
	}

	// float32 forward passes cap the precision, so the step and tolerance
	// are coarser than a float64 check would use.
	const h = 1e-2
	for i := range param.Data {
		orig := param.Data[i]
		param.Data[i] = orig + h
		lp, _ := build()
		param.Data[i] = orig - h
		lm, _ := build()
		param.Data[i] = orig
		num := float64(lp.Value.Data[0]-lm.Value.Data[0]) / (2 * h)
		if math.Abs(num-analytic[i]) > 2e-2*(1+math.Abs(num)) {
			t.Errorf("%s grad %d: numeric=%.8f analytic=%.8f", name, i, num, analytic[i])
		}
	}
}

func TestAutogradOps(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	w := tensai.RandomMatrix(3, 4, rng).Tensor()
	x := tensai.RandomMatrix(5, 3, rng).Tensor()
	b := tensai.RandomMatrix(1, 4, rng).Tensor()
	other := tensai.RandomMatrix(5, 4, rng).Tensor()

	cases := []struct {
		name  string
		param *tensai.Tensor
		build func() (*Node, *Node)
	}{
		{"matmul", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).Sum(), p
		}},
		{"addrow", b, func() (*Node, *Node) {
			p := Param(b)
			return Input(x).MatMul(Input(w)).AddRow(p).Tanh().Sum(), p
		}},
		{"add-sub-mul", w, func() (*Node, *Node) {
			p := Param(w)
			y := Input(x).MatMul(p)
			return y.Add(Input(other)).Sub(Input(other).Scale(0.5)).MulElem(y).Mean(), p
		}},
		{"relu-sigmoid", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).ReLU().Sigmoid().Sum(), p
		}},
		{"mse", w, func() (*Node, *Node) {
			p := Param(w)
			return Input(x).MatMul(p).MSELoss(other), p
		}},
	}
	for _, tc := range cases {
		checkParamGrad(t, tc.param, tc.build, tc.name)
	}
}

func TestAutogradSoftmaxCE(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	w := tensai.RandomMatrix(3, 4, rng).Tensor()
	x := tensai.RandomMatrix(6, 3, rng).Tensor()
	target := tensai.NewMatrix(6, 1).Tensor()
	for i := range target.Data {
		target.Data[i] = tensai.Float(rng.Intn(4))
	}
	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).SoftmaxCELoss(target), p
	}, "softmax-ce")
}

func TestAutogradSharedParam(t *testing.T) {
	// A parameter used twice must accumulate both contributions.
	w := tensai.NewTensor(2, 2)
	w.Data = []tensai.Float{1, 2, 3, 4}
	x := tensai.NewTensor(1, 2)
	x.Data = []tensai.Float{1, 1}
	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		h := Input(x).MatMul(p)
		return h.MatMul(p).Sum(), p
	}, "shared")
}

func TestAutogradTrainsXOR(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	inputs, err := tensai.NewMatrixFromSlice(4, 2, []tensai.Float{0, 0, 0, 1, 1, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	targets, err := tensai.NewMatrixFromSlice(4, 1, []tensai.Float{0, 1, 1, 0})
	if err != nil {
		t.Fatal(err)
	}

	w1 := Param(tensai.RandomMatrix(2, 8, rng))
	b1 := Param(tensai.NewMatrix(1, 8))
	w2 := Param(tensai.RandomMatrix(8, 1, rng))
	b2 := Param(tensai.NewMatrix(1, 1))
	trainer := NewTrainer(optim.NewAdam(0.05), w1, b1, w2, b2)

	forward := func(x *tensai.Matrix) *Node {
		return Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).AddRow(b2).Sigmoid()
	}

	var lossVal tensai.Float
	for step := 0; step < 2000; step++ {
		lossVal = trainer.Step(forward(inputs).MSELoss(targets.Tensor()))
	}
	if lossVal > 0.01 {
		t.Fatalf("autograd XOR failed to converge: loss=%g", lossVal)
	}
	pred := forward(inputs)
	for r := 0; r < 4; r++ {
		got := pred.Value.At(r, 0)
		want := targets.At(r, 0)
		if math.Abs(float64(got-want)) > 0.15 {
			t.Errorf("sample %d: predicted %.3f, want %g", r, got, want)
		}
	}
}

func TestToDot(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	x := Input(tensai.RandomMatrix(3, 2, rng)).Named("x")
	w := Param(tensai.RandomMatrix(2, 4, rng)).Named("w")
	root := x.MatMul(w).Tanh().Sum()
	dot := root.ToDot()

	for _, want := range []string{
		"digraph tensai",
		`"x\n3x2"`, `"w\n2x4"`,
		`"matmul\n3x4"`, `"tanh\n3x4"`, `"sum\n1x1"`,
	} {
		if !strings.Contains(dot, want) {
			t.Errorf("ToDot output missing %q:\n%s", want, dot)
		}
	}
	// x->matmul, w->matmul, matmul->tanh, tanh->sum
	if got := strings.Count(dot, "->"); got != 4 {
		t.Errorf("expected 4 edges, got %d:\n%s", got, dot)
	}
}

func TestAutogradTransposeAndSoftmax(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	w := tensai.RandomMatrix(3, 4, rng).Tensor()
	x := tensai.RandomMatrix(5, 3, rng).Tensor()

	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).T().MatMul(Input(x)).Sum(), p
	}, "transpose")

	// Weighted sum keeps the softmax gradient non-trivial (a plain Sum of
	// each row is constant 1 and would zero it out).
	weights := tensai.RandomMatrix(5, 4, rng).Tensor()
	checkParamGrad(t, w, func() (*Node, *Node) {
		p := Param(w)
		return Input(x).MatMul(p).Softmax().MulElem(Input(weights)).Sum(), p
	}, "softmax")
}
