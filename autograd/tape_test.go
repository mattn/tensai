package autograd

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/optim"
)

// buildXOR returns a training step for the XOR network, along with its
// parameters, so the same model can run with and without a tape.
func buildXOR(t *testing.T, seed int64) (params []*Node, step func() tensai.Float) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
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
	params = []*Node{w1, b1, w2, b2}
	trainer := NewTrainer(optim.NewAdam(0.05), params...)
	step = func() tensai.Float {
		out := Input(inputs).MatMul(w1).AddRow(b1).Tanh().MatMul(w2).AddRow(b2).Sigmoid()
		return trainer.Step(out.MSELoss(targets.Tensor()))
	}
	return params, step
}

// TestTapeMatchesUntaped trains the same model twice, once with a tape and
// once without: recycled buffers must not change a single weight.
func TestTapeMatchesUntaped(t *testing.T) {
	plain, plainStep := buildXOR(t, 5)
	taped, tapedStep := buildXOR(t, 5)
	tape := NewTape()
	tape.Bind(taped...)

	var plainLoss, tapedLoss tensai.Float
	for i := 0; i < 300; i++ {
		plainLoss = plainStep()
		tapedLoss = tapedStep()
		tape.Reset()
	}
	if plainLoss != tapedLoss {
		t.Fatalf("loss differs: plain %g, taped %g", plainLoss, tapedLoss)
	}
	if plainLoss > 0.05 {
		t.Fatalf("XOR did not converge: loss=%g", plainLoss)
	}
	for i := range plain {
		for j, want := range plain[i].Value.Data {
			if got := taped[i].Value.Data[j]; got != want {
				t.Fatalf("param %d element %d: taped %g, plain %g", i, j, got, want)
			}
		}
	}
}

// buildWide returns a training step for a network whose buffers dominate
// its bookkeeping, which is where a tape pays.
func buildWide(t *testing.T, seed int64) (params []*Node, step func() tensai.Float) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	x := tensai.RandomMatrix(32, 64, rng)
	y := tensai.RandomMatrix(32, 16, rng).Tensor()
	w1 := Param(tensai.RandomMatrix(64, 128, rng))
	b1 := Param(tensai.NewMatrix(1, 128))
	w2 := Param(tensai.RandomMatrix(128, 16, rng))
	params = []*Node{w1, b1, w2}
	trainer := NewTrainer(optim.NewAdam(0.01), params...)
	step = func() tensai.Float {
		out := Input(x).MatMul(w1).AddRow(b1).Tanh().MatMul(w2)
		return trainer.Step(out.MSELoss(y))
	}
	return params, step
}

// TestTapeReusesBuffers checks that a reset tape hands the same memory back
// out instead of allocating again.
func TestTapeReusesBuffers(t *testing.T) {
	_, plainStep := buildWide(t, 9)
	taped, tapedStep := buildWide(t, 9)
	tape := NewTape()
	tape.Bind(taped...)

	tapedStep() // the first step fills the pool
	tape.Reset()

	// Bytes, not object counts, are what the tape is for: the node and
	// closure bookkeeping still allocates either way, but the buffers those
	// nodes hold should stop coming from the heap.
	plainBytes := bytesPerRun(20, func() { plainStep() })
	tapedBytes := bytesPerRun(20, func() {
		tapedStep()
		tape.Reset()
	})
	if tapedBytes >= plainBytes/4 {
		t.Errorf("tape did not cut allocation: %d B/step with, %d B/step without", tapedBytes, plainBytes)
	}
	t.Logf("%d B/step with the tape, %d B/step without", tapedBytes, plainBytes)

	// The buffers themselves must come back, not just their sizes: the
	// value of the same node in two successive steps shares one array.
	w := Param(tensai.RandomMatrix(3, 4, rand.New(rand.NewSource(3))))
	tape2 := NewTape()
	tape2.Bind(w)
	x := tensai.RandomMatrix(2, 3, rand.New(rand.NewSource(4)))
	first := Input(x).MatMul(w)
	tape2.Reset()
	second := Input(x).MatMul(w)
	if &first.Value.Data[0] != &second.Value.Data[0] {
		t.Error("second step allocated a fresh buffer instead of reusing the first")
	}
}

// TestTapeLeavesParamsAlone checks that binding a tape never recycles a
// parameter's own value, which has to survive every Reset.
func TestTapeLeavesParamsAlone(t *testing.T) {
	w := Param(tensai.NewTensor(2, 2))
	copy(w.Value.Data, []tensai.Float{1, 2, 3, 4})
	tape := NewTape()
	tape.Bind(w)
	before := w.Value.Data

	for i := 0; i < 5; i++ {
		w.Mul(w).Sum().Backward()
		ZeroGrads(w)
		tape.Reset()
	}
	if &before[0] != &w.Value.Data[0] {
		t.Fatal("parameter value was replaced")
	}
	for i, want := range []tensai.Float{1, 2, 3, 4} {
		if w.Value.Data[i] != want {
			t.Fatalf("parameter element %d = %g, want %g", i, w.Value.Data[i], want)
		}
	}
}

// TestTapeGradientsStayCorrect runs the numeric-gradient check on a taped
// graph, so a recycled buffer that still held stale numbers would show up.
func TestTapeGradientsStayCorrect(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	x := randTensor(rng, 2, 3, 4)
	w := randTensor(rng, 4, 5)
	weights := randTensor(rng, 2, 3, 5)
	gain := randTensor(rng, 5)
	tape := NewTape()

	build := func() (*Node, *Node) {
		p := Param(w)
		tape.Bind(p)
		out := Input(x).MatMul(p).LayerNorm(Input(gain), nil, 1e-5).GELU()
		return out.Mul(Input(weights)).Sum(), p
	}
	// Run once and reset, so the check below runs on recycled buffers.
	loss, p := build()
	loss.Backward()
	ZeroGrads(p)
	tape.Reset()

	checkParamGrad(t, w, build, "taped-graph")

	// A scalar loss must still come out identical after a reset.
	tape.Reset()
	first, _ := build()
	got := first.Scalar()
	tape.Reset()
	second, _ := build()
	if math.Abs(float64(got-second.Scalar())) > 1e-6 {
		t.Errorf("loss changed across a reset: %g then %g", got, second.Scalar())
	}
}

// bytesPerRun reports the heap bytes one call of f allocates, averaged over
// n runs.
func bytesPerRun(n int, f func()) uint64 {
	var before, after runtime.MemStats
	f() // warm up, so first-run growth is not counted
	runtime.ReadMemStats(&before)
	for i := 0; i < n; i++ {
		f()
	}
	runtime.ReadMemStats(&after)
	return (after.TotalAlloc - before.TotalAlloc) / uint64(n)
}
