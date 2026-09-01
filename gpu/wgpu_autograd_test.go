//go:build (wgpu || wgpu24) && (linux || darwin || windows)

package gpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

// TestGPUAcceleratesAutograd runs one forward and backward pass of a small
// network twice -- once on the CPU, once with the device installed as the
// accelerator -- and checks that every parameter gradient agrees. The
// backward pass uses both transposed products, so this covers the whole
// routing path end to end.
func TestGPUAcceleratesAutograd(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	grads := func(seed int64) []*tensai.Tensor {
		rng := rand.New(rand.NewSource(seed))
		x := randTensor(rng, 32, 48)
		y := randTensor(rng, 32, 16)
		w1 := autograd.Param(randTensor(rng, 48, 64))
		w2 := autograd.Param(randTensor(rng, 64, 16))
		loss := autograd.Input(x).MatMul(w1).Tanh().MatMul(w2).MSELoss(y)
		loss.Backward()
		return []*tensai.Tensor{w1.Grad(), w2.Grad()}
	}

	want := grads(19)
	// A threshold of zero sends even these small products to the device.
	prev := tensai.UseAcceleratorThreshold(g, 0)
	got := grads(19)
	tensai.UseAccelerator(prev)

	for i := range want {
		if len(got[i].Data) != len(want[i].Data) {
			t.Fatalf("gradient %d: got %v want %v", i, got[i].Shape, want[i].Shape)
		}
		for j := range want[i].Data {
			diff := math.Abs(float64(got[i].Data[j] - want[i].Data[j]))
			if diff > 1e-4*(1+math.Abs(float64(want[i].Data[j]))) {
				t.Fatalf("gradient %d element %d: gpu=%v cpu=%v", i, j, got[i].Data[j], want[i].Data[j])
			}
		}
	}
}

// TestGPUAcceleratedTrainingConverges checks that a network trained with
// every product on the device still learns.
func TestGPUAcceleratedTrainingConverges(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(23))
	x := randTensor(rng, 64, 32)
	target := randTensor(rng, 64, 8)
	w1 := autograd.Param(randTensor(rng, 32, 64))
	w2 := autograd.Param(randTensor(rng, 64, 8))
	trainer := autograd.NewTrainer(optim.NewAdam(0.01), w1, w2)
	tape := autograd.NewTape()
	tape.Bind(w1, w2)

	prev := tensai.UseAcceleratorThreshold(g, 0)
	defer tensai.UseAccelerator(prev)

	var first, last tensai.Float
	for step := 0; step < 60; step++ {
		loss := autograd.Input(x).MatMul(w1).Tanh().MatMul(w2).MSELoss(target)
		last = trainer.Step(loss)
		tape.Reset()
		if step == 0 {
			first = last
		}
	}
	if !(last < first*0.5) {
		t.Fatalf("loss did not fall on the device: %g -> %g", first, last)
	}
}

// TestGPUResidentTraining trains the same MLP twice from the same
// initialization -- once on the CPU, once with the graph resident on the
// device -- and checks the two stay together. With the tape on a device
// the values, the gradients and the Adam update all stay there; only the
// loss comes home each step.
func TestGPUResidentTraining(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	build := func() (x, y *tensai.Tensor, params []*autograd.Node) {
		rng := rand.New(rand.NewSource(101))
		x = randTensor(rng, 32, 24)
		y = randTensor(rng, 32, 8)
		w1 := autograd.Param(randTensor(rng, 24, 32))
		b1 := autograd.Param(tensai.NewTensor(1, 32))
		w2 := autograd.Param(randTensor(rng, 32, 8))
		return x, y, []*autograd.Node{w1, b1, w2}
	}
	forward := func(x *tensai.Tensor, p []*autograd.Node) *autograd.Node {
		return autograd.Input(x).MatMul(p[0]).Add(p[1]).Tanh().MatMul(p[2])
	}

	const steps = 120
	x, y, cpuP := build()
	cpuTrainer := autograd.NewTrainer(optim.NewAdam(0.01), cpuP...)
	var firstLoss, cpuLoss tensai.Float
	for i := 0; i < steps; i++ {
		cpuLoss = cpuTrainer.Step(forward(x, cpuP).MSELoss(y))
		if i == 0 {
			firstLoss = cpuLoss
		}
	}

	_, _, devP := build()
	devTrainer := autograd.NewTrainer(optim.NewAdam(0.01), devP...)
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(devP...)
	var devLoss tensai.Float
	for i := 0; i < steps; i++ {
		devLoss = devTrainer.Step(forward(x, devP).MSELoss(y))
		tape.Reset()
	}

	if diff := math.Abs(float64(cpuLoss - devLoss)); diff > 1e-2*(1+math.Abs(float64(cpuLoss))) {
		t.Fatalf("loss differs after %d steps: cpu=%g device=%g", steps, cpuLoss, devLoss)
	}
	for i := range cpuP {
		want, got := cpuP[i].Value(), devP[i].Value()
		for j := range want.Data {
			if diff := math.Abs(float64(want.Data[j] - got.Data[j])); diff > 1e-2*(1+math.Abs(float64(want.Data[j]))) {
				t.Fatalf("param %d element %d: cpu=%v device=%v", i, j, want.Data[j], got.Data[j])
			}
		}
	}
	if !(cpuLoss < firstLoss*0.5) {
		t.Fatalf("training did not progress: %g -> %g", firstLoss, cpuLoss)
	}
}

// TestGPUResidentStaysOnDevice checks that a resident graph really keeps
// its intermediates on the GPU: nothing but the loss should have a host
// copy after a step.
func TestGPUResidentStaysOnDevice(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	rng := rand.New(rand.NewSource(103))
	x := randTensor(rng, 16, 16)
	w := autograd.Param(randTensor(rng, 16, 16))
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(w)

	// The input carries no tape of its own, so this also checks that a
	// per-step constant joins the graph's rather than dropping it to the
	// CPU for the rest of the step.
	prod := autograd.Input(x).MatMul(w)
	if !prod.Resident() {
		t.Fatal("the product did not stay on the device")
	}
	h := prod.Tanh()
	if !h.Resident() {
		t.Fatal("the activation did not stay on the device")
	}
	// Reading it brings it home, and the value must match the CPU.
	got := h.Value()
	want, err := tensai.MatMul(x, w.Value())
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		w := math.Tanh(float64(want.Data[i]))
		if diff := math.Abs(float64(got.Data[i]) - w); diff > 1e-4 {
			t.Fatalf("element %d: device=%v cpu=%v", i, got.Data[i], w)
		}
	}
	tape.Reset()
}

// TestGPUResidentTransformer trains a transformer block -- embeddings, a
// pre-norm attention with a causal mask, a GELU feed-forward, and
// cross-entropy over the vocabulary -- on the CPU and on the device, and
// checks the two agree. Every op but the embedding lookup has a device
// kernel, so this exercises LayerNorm, softmax, the permutes attention
// needs, and the products between them.
func TestGPUResidentTransformer(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	const (
		batch, seq, model, heads, vocab = 2, 8, 32, 4, 11
		headDim                         = model / heads
	)
	build := func() (tokens, labels []int, mask *tensai.Tensor, params []*autograd.Node) {
		rng := rand.New(rand.NewSource(211))
		tokens = make([]int, batch*seq)
		labels = make([]int, batch*seq)
		for i := range tokens {
			tokens[i] = rng.Intn(vocab)
			labels[i] = rng.Intn(vocab)
		}
		mask = tensai.NewTensor(1, 1, seq, seq)
		for i := 0; i < seq; i++ {
			for j := i + 1; j < seq; j++ {
				mask.Data[i*seq+j] = tensai.Float(math.Inf(-1))
			}
		}
		ones := tensai.NewTensor(model)
		for i := range ones.Data {
			ones.Data[i] = 1
		}
		params = []*autograd.Node{
			autograd.Param(randTensor(rng, vocab, model)), // embeddings
			autograd.Param(ones),                          // norm gain
			autograd.Param(tensai.NewTensor(model)),       // norm bias
			autograd.Param(randTensor(rng, model, model)), // wq
			autograd.Param(randTensor(rng, model, model)), // wk
			autograd.Param(randTensor(rng, model, model)), // wv
			autograd.Param(randTensor(rng, model, model)), // wo
			autograd.Param(randTensor(rng, model, vocab)), // head
		}
		return tokens, labels, mask, params
	}
	forward := func(tokens []int, mask *tensai.Tensor, p []*autograd.Node) *autograd.Node {
		heads3 := func(t *autograd.Node) *autograd.Node {
			return t.Reshape(batch, seq, heads, headDim).Transpose(0, 2, 1, 3)
		}
		x := p[0].Embed(tokens, batch, seq)
		h := x.LayerNorm(p[1], p[2], 1e-5)
		q, k, v := heads3(h.MatMul(p[3])), heads3(h.MatMul(p[4])), heads3(h.MatMul(p[5]))
		att := q.MatMul(k.T()).Scale(1 / float32(math.Sqrt(headDim))).Add(autograd.Input(mask)).Softmax()
		y := att.MatMul(v).Transpose(0, 2, 1, 3).Reshape(batch, seq, model).MatMul(p[6])
		return x.Add(y).GELU().MatMul(p[7])
	}

	const steps = 15
	tokens, labels, mask, cpuP := build()
	cpuTrainer := autograd.NewTrainer(optim.NewAdam(0.01), cpuP...)
	var cpuLoss tensai.Float
	for i := 0; i < steps; i++ {
		cpuLoss = cpuTrainer.Step(forward(tokens, mask, cpuP).CrossEntropy(labels))
	}

	_, _, _, devP := build()
	devTrainer := autograd.NewTrainer(optim.NewAdam(0.01), devP...)
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(devP...)
	var devLoss tensai.Float
	for i := 0; i < steps; i++ {
		logits := forward(tokens, mask, devP)
		if i == 0 && !logits.Resident() {
			t.Fatal("the block did not stay on the device: something pulled a value home")
		}
		devLoss = devTrainer.Step(logits.CrossEntropy(labels))
		tape.Reset()
	}

	if diff := math.Abs(float64(cpuLoss - devLoss)); diff > 1e-2*(1+math.Abs(float64(cpuLoss))) {
		t.Fatalf("loss differs after %d steps: cpu=%g device=%g", steps, cpuLoss, devLoss)
	}
	names := []string{"embed", "gain", "bias", "wq", "wk", "wv", "wo", "head"}
	for i := range cpuP {
		want, got := cpuP[i].Value(), devP[i].Value()
		for j := range want.Data {
			if diff := math.Abs(float64(want.Data[j] - got.Data[j])); diff > 2e-2*(1+math.Abs(float64(want.Data[j]))) {
				t.Fatalf("%s element %d: cpu=%v device=%v", names[i], j, want.Data[j], got.Data[j])
			}
		}
	}
}

// TestGPUResidentShapes pins the shapes a resident chain reports. A kernel
// may hand back a flatter view of the same elements -- the embedding lookup
// returns (rows, dim) for what the graph calls (batch, seq, dim) -- and the
// next device op reads the buffer's shape, so a mismatch quietly reshapes
// everything downstream. It surfaced as a broadcast failure two operations
// later, which is why the shapes are checked here step by step.
func TestGPUResidentShapes(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	const batch, seq, dim, vocab = 2, 4, 8, 5
	rng := rand.New(rand.NewSource(311))
	table := autograd.Param(randTensor(rng, vocab, dim))
	pos := autograd.Param(randTensor(rng, 1, seq, dim))
	w := autograd.Param(randTensor(rng, dim, dim))
	tape := autograd.NewTape()
	tape.UseDevice(g)
	tape.Bind(table, pos, w)
	defer tape.Reset()

	tokens := make([]int, batch*seq)
	for i := range tokens {
		tokens[i] = rng.Intn(vocab)
	}
	steps := []struct {
		name string
		node *autograd.Node
		want []int
	}{}
	x := table.Embed(tokens, batch, seq)
	steps = append(steps, struct {
		name string
		node *autograd.Node
		want []int
	}{"embed", x, []int{batch, seq, dim}})
	h := x.Add(pos)
	steps = append(steps, struct {
		name string
		node *autograd.Node
		want []int
	}{"add", h, []int{batch, seq, dim}})
	y := h.MatMul(w)
	steps = append(steps, struct {
		name string
		node *autograd.Node
		want []int
	}{"matmul", y, []int{batch, seq, dim}})
	sum := x.Add(y) // the residual that fails when a shape drifted
	steps = append(steps, struct {
		name string
		node *autograd.Node
		want []int
	}{"residual", sum, []int{batch, seq, dim}})

	for _, s := range steps {
		if !s.node.Resident() {
			t.Errorf("%s left the device", s.name)
		}
		got := s.node.Shape()
		if len(got) != len(s.want) {
			t.Fatalf("%s shape %v, want %v", s.name, got, s.want)
		}
		for i := range s.want {
			if got[i] != s.want[i] {
				t.Fatalf("%s shape %v, want %v", s.name, got, s.want)
			}
		}
		if v := s.node.Value(); len(v.Shape) != len(s.want) {
			t.Fatalf("%s downloaded shape %v, want %v", s.name, v.Shape, s.want)
		}
	}
}

// TestGPUSequentialGraph trains a Sequential model through its graph form
// on the device and checks the result against the same training on the
// host. The model is an ordinary layer stack: it is the graph that puts it
// on a GPU, which the hand-written Forward and Backward cannot.
func TestGPUSequentialGraph(t *testing.T) {
	g := openTestGPU(t)
	defer g.Close()

	build := func() (*model.Sequential, *tensai.Matrix, *tensai.Matrix) {
		rng := rand.New(rand.NewSource(313))
		x := randTensor(rng, 16, 24)
		y := randTensor(rng, 16, 4)
		xm, err := x.Matrix()
		if err != nil {
			t.Fatal(err)
		}
		ym, err := y.Matrix()
		if err != nil {
			t.Fatal(err)
		}
		net := model.NewSequential()
		net.Add(layer.NewDense(32))
		net.Add(&layer.Tanh{})
		net.Add(layer.NewDense(4))
		// NewSequential seeds its own generator, so two models built the
		// same way start from the same weights.
		if err := net.Compile(24, loss.MeanSquaredError{}, optim.NewAdam(0.01)); err != nil {
			t.Fatal(err)
		}
		return net, xm, ym
	}

	train := func(dev *gpu.Device) (*model.Sequential, tensai.Float) {
		net, x, y := build()
		graph, err := net.Graph()
		if err != nil {
			t.Fatal(err)
		}
		trainer := autograd.NewTrainer(optim.NewAdam(0.01), graph.Params()...)
		tape := autograd.NewTape()
		if dev != nil {
			tape.UseDevice(dev)
		}
		tape.Bind(graph.Params()...)
		var last tensai.Float
		for i := 0; i < 60; i++ {
			l, err := graph.Loss(graph.Forward(autograd.Input(x)), y)
			if err != nil {
				t.Fatal(err)
			}
			last = trainer.Step(l)
			tape.Reset()
		}
		graph.Sync()
		return net, last
	}

	cpuNet, cpuLoss := train(nil)
	devNet, devLoss := train(g)
	if diff := math.Abs(float64(cpuLoss - devLoss)); diff > 1e-3*(1+math.Abs(float64(cpuLoss))) {
		t.Fatalf("loss differs: cpu=%g device=%g", cpuLoss, devLoss)
	}

	// Sync copied the device's weights back into the layers, so the model
	// predicts with them.
	_, x, _ := build()
	want, err := cpuNet.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	got, err := devNet.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Data {
		if diff := math.Abs(float64(want.Data[i] - got.Data[i])); diff > 1e-3*(1+math.Abs(float64(want.Data[i]))) {
			t.Fatalf("prediction %d: cpu=%v device=%v", i, want.Data[i], got.Data[i])
		}
	}
}
