package model

import (
	"fmt"
	"math/rand"

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
// Every layer has a graph form. Dropout and BatchNorm behave as they do in
// the layer stack, and SetTraining switches between the two halves of that
// behaviour.
type Graph struct {
	steps    []func(*autograd.Node) *autograd.Node
	params   []*autograd.Node
	hosts    []hostParam
	loss     string
	rng      *rand.Rand
	training bool
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
	g := &Graph{loss: s.lossName, rng: s.rng, training: true}
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

// scalar wraps one value as a tensor the element-wise ops broadcast.
func scalar(v tensai.Float) *tensai.Tensor {
	t := tensai.NewTensor(1)
	t.Data[0] = v
	return t
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

	case *layer.Embedding:
		w, _ := v.Params()
		table := g.param(w.Tensor())
		vocab, dim := w.Rows, w.Cols
		return func(x *autograd.Node) *autograd.Node {
			// The ids arrive as an ordinary (batch, tokens) matrix of
			// whole numbers, which the lookup wants as indices.
			shape := x.Shape()
			if len(shape) != 2 {
				panic(fmt.Sprintf("tensai: embedding needs a (batch, tokens) input, got %v", shape))
			}
			batch, tokens := shape[0], shape[1]
			ids := make([]int, batch*tokens)
			for i, f := range x.Value().Data {
				id := int(f)
				if tensai.Float(id) != f || id < 0 || id >= vocab {
					panic(fmt.Sprintf("tensai: embedding token id %g out of range [0,%d)", f, vocab))
				}
				ids[i] = id
			}
			// One row per token, then the row's tokens laid end to end,
			// which is the layout the layer produces.
			return table.Embed(ids, batch, tokens).Reshape(batch, tokens*dim)
		}, nil

	case *layer.Dropout:
		rate := v.Rate
		return func(x *autograd.Node) *autograd.Node {
			if !g.training {
				return x
			}
			return x.Dropout(rate, g.rng)
		}, nil

	case *layer.BatchNorm:
		gamma, beta := v.Params()
		gn := g.param(gamma.Tensor())
		bn := g.param(rowVector(beta))
		mean, variance := v.RunningStats()
		eps, momentum := v.Eps, v.Momentum
		epsNode := autograd.Input(scalar(eps))
		return func(x *autograd.Node) *autograd.Node {
			if !g.training {
				// Inference normalizes with the running estimates, which
				// are data rather than parameters: no gradient flows to
				// them.
				sd := autograd.Input(rowVector(variance)).Add(epsNode).Sqrt()
				return x.Sub(autograd.Input(rowVector(mean))).Div(sd).Mul(gn).Add(bn)
			}
			// Training normalizes with this batch's own statistics, and
			// the gradient flows through them -- which is the whole point
			// of the layer, and comes out of the graph for free.
			m := x.MeanAxis(0, true)
			d := x.Sub(m)
			varNode := d.Mul(d).MeanAxis(0, true)
			out := d.Div(varNode.Add(epsNode).Sqrt()).Mul(gn).Add(bn)
			// The running estimates are a side effect of the forward pass,
			// exactly as in layer.BatchNorm.
			mv, vv := m.Value().Data, varNode.Value().Data
			for i := range mean {
				mean[i] = momentum*mean[i] + (1-momentum)*mv[i]
				variance[i] = momentum*variance[i] + (1-momentum)*vv[i]
			}
			return out
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

// SetTraining switches Dropout and BatchNorm between their training and
// inference behaviour, the way Fit and Predict do for the layer stack. A
// graph starts out training.
func (g *Graph) SetTraining(v bool) { g.training = v }

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
