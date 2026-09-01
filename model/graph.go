package model

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
	"github.com/mattn/tensai/layer"
)

// Graph is a Sequential built as an autograd graph instead of a stack of
// hand-written Forward and Backward passes. The parameters are the very
// matrices the layers hold, so training through the graph trains the model
// -- Predict, Save and the rest keep working afterwards -- and the graph
// can run on a GPU, which the layer stack cannot:
//
//	g, err := net.Graph()
//	trainer := autograd.NewTrainer(optim.NewAdam(0.01), g.Params()...)
//	tape := autograd.NewTape()
//	tape.UseDevice(dev) // optional
//	tape.Bind(g.Params()...)
//	for step := 0; step < steps; step++ {
//		trainer.Step(g.Loss(g.Forward(autograd.Input(x)), y))
//		tape.Reset()
//	}
//	g.Sync() // only needed after training on a device
//
// Dropout, BatchNorm and Embedding have no graph form yet, so a model
// holding one is refused rather than silently trained differently.
type Graph struct {
	steps  []func(*autograd.Node) *autograd.Node
	params []*autograd.Node
	hosts  []hostParam
	loss   string
}

// hostParam remembers where a parameter lives in its layer, so a value
// that came back from a device can be copied home.
type hostParam struct {
	node *autograd.Node
	data []tensai.Float
}

// Graph builds the graph form of a compiled model.
func (s *Sequential) Graph() (*Graph, error) {
	if len(s.layers) == 0 {
		return nil, fmt.Errorf("tensai: cannot build a graph from a model with no layers")
	}
	g := &Graph{loss: s.lossName}
	for i, l := range s.layers {
		step, err := g.add(l)
		if err != nil {
			return nil, fmt.Errorf("tensai: layer %d: %w", i, err)
		}
		g.steps = append(g.steps, step)
	}
	return g, nil
}

// param registers a trainable buffer, sharing the layer's memory so an
// update lands where the layer will read it.
func (g *Graph) param(t *tensai.Tensor) *autograd.Node {
	n := autograd.Param(t)
	g.params = append(g.params, n)
	g.hosts = append(g.hosts, hostParam{node: n, data: t.Data})
	return n
}

// rowVector views a bias slice as a 1 x n matrix without copying it.
func rowVector(b []tensai.Float) *tensai.Tensor {
	return &tensai.Tensor{Shape: []int{1, len(b)}, Data: b}
}

// add turns one layer into a step of the graph.
func (g *Graph) add(l layer.Layer) (func(*autograd.Node) *autograd.Node, error) {
	switch v := l.(type) {
	case *layer.Dense:
		w, b := v.Params()
		wn, bn := g.param(w.Tensor()), g.param(rowVector(b))
		return func(x *autograd.Node) *autograd.Node { return x.MatMul(wn).Add(bn) }, nil

	case *layer.LayerNorm:
		gamma, beta := v.Params()
		gn := g.param(&tensai.Tensor{Shape: []int{gamma.Cols}, Data: gamma.Data})
		bn := g.param(&tensai.Tensor{Shape: []int{len(beta)}, Data: beta})
		eps := v.Eps()
		return func(x *autograd.Node) *autograd.Node { return x.LayerNorm(gn, bn, eps) }, nil

	case *layer.Conv2D:
		inH, inW, inC, outC, kernel, stride, pad := v.Shape()
		w, b := v.Params()
		wn, bn := g.param(w.Tensor()), g.param(&tensai.Tensor{Shape: []int{len(b)}, Data: b})
		cfg := autograd.Conv{Kernel: kernel, Stride: stride, Pad: pad}
		outH, outW := (inH+2*pad-kernel)/stride+1, (inW+2*pad-kernel)/stride+1
		return func(x *autograd.Node) *autograd.Node {
			// The rows a Sequential passes around are channel-major
			// images; the graph works on them with the spatial axes split
			// out, and folds them back for the next layer.
			img := x.Reshape(-1, inC, inH, inW)
			return img.Conv2D(wn, bn, cfg).Reshape(-1, outC*outH*outW)
		}, nil

	case *layer.MaxPool2D:
		inH, inW, ch, size := v.Shape()
		outH, outW := inH/size, inW/size
		return func(x *autograd.Node) *autograd.Node {
			img := x.Reshape(-1, ch, inH, inW)
			return img.MaxPool2D(size).Reshape(-1, ch*outH*outW)
		}, nil

	case *layer.ReLU:
		return (*autograd.Node).ReLU, nil
	case *layer.Sigmoid:
		return (*autograd.Node).Sigmoid, nil
	case *layer.Tanh:
		return (*autograd.Node).Tanh, nil
	case *layer.GELU:
		return (*autograd.Node).GELU, nil
	case *layer.Softmax:
		return (*autograd.Node).Softmax, nil
	case *layer.LeakyReLU:
		alpha := v.Alpha
		return func(x *autograd.Node) *autograd.Node { return x.LeakyReLU(alpha) }, nil
	}
	return nil, fmt.Errorf("%T has no graph form", l)
}

// Params returns the trainable nodes, in layer order, for a Trainer and a
// Tape.
func (g *Graph) Params() []*autograd.Node { return g.params }

// Forward runs the stack on a (batch, inputs) node.
func (g *Graph) Forward(x *autograd.Node) *autograd.Node {
	for _, step := range g.steps {
		x = step(x)
	}
	return x
}

// Loss applies the loss the model was compiled with. Softmax
// cross-entropy takes an Mx1 matrix of class indices, as it does
// everywhere else here; the others take a target the shape of the output.
func (g *Graph) Loss(pred *autograd.Node, target *tensai.Matrix) (*autograd.Node, error) {
	switch g.loss {
	case "softmax_ce":
		return pred.SoftmaxCELoss(target.Tensor()), nil
	case "mse":
		return pred.MSELoss(target.Tensor()), nil
	case "":
		return nil, fmt.Errorf("tensai: the model has no loss; compile it first")
	}
	return nil, fmt.Errorf("tensai: loss %q has no graph form", g.loss)
}

// Sync copies the parameters back into the layers. Training on the host
// updates them in place and needs no call; a device holds its own copies,
// and this is what brings them home.
func (g *Graph) Sync() {
	for _, h := range g.hosts {
		v := h.node.Value()
		if len(v.Data) != len(h.data) {
			continue
		}
		if &v.Data[0] != &h.data[0] {
			copy(h.data, v.Data)
		}
	}
}
