package model

import (
	"fmt"
	"math/rand"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/optim"
)

// Model is a trained, ready-to-predict network.
type Model interface {
	Predict(input *tensai.Matrix) (*tensai.Matrix, error)
}

// Sequential stacks layers and runs forward/backward passes.
type Sequential struct {
	layers    []layer.Layer
	optimizer optim.Optimizer
	loss      loss.Loss
	lossName  string
	rng       *rand.Rand
	lossGrad  *tensai.Matrix
}

type lossInto interface {
	LossInto(pred, target, grad *tensai.Matrix) (tensai.Float, error)
}

// NewSequential returns an empty Sequential model. optimizer and loss are
// configured via Compile.
func NewSequential() *Sequential {
	return &Sequential{rng: rand.New(rand.NewSource(0))}
}

// Add appends a layer to the network. Layers are added in forward order.
func (s *Sequential) Add(layer layer.Layer) *Sequential {
	s.layers = append(s.layers, layer)
	return s
}

// Compile wires the loss and optimizer and initializes all parameters.
// inputCols is the number of features in a single input row. Models that
// start from flattened images should use CompileImage instead, which also
// threads the spatial shape through the stack.
func (s *Sequential) Compile(inputCols int, loss loss.Loss, optimizer optim.Optimizer) error {
	return s.compile(inputCols, layer.Image{}, loss, optimizer)
}

// CompileImage is Compile for models whose input rows are flattened
// channel-major images. The spatial shape threads through the stack:
// Conv2D and MaxPool2D take their input dimensions from it, and layers
// that keep the row width (activations, BatchNorm, LayerNorm, Dropout)
// pass it along, so only the model input states its geometry.
func (s *Sequential) CompileImage(in layer.Image, loss loss.Loss, optimizer optim.Optimizer) error {
	if in.H <= 0 || in.W <= 0 || in.C <= 0 {
		return fmt.Errorf("tensai: CompileImage needs a positive input shape, got %dx%dx%d", in.H, in.W, in.C)
	}
	return s.compile(in.Cols(), in, loss, optimizer)
}

func (s *Sequential) compile(inputCols int, img layer.Image, loss loss.Loss, optimizer optim.Optimizer) error {
	if len(s.layers) == 0 {
		return fmt.Errorf("tensai: cannot compile a model with no layers")
	}
	if loss == nil {
		return fmt.Errorf("tensai: nil loss")
	}
	if optimizer == nil {
		return fmt.Errorf("tensai: nil optimizer")
	}
	s.loss = loss
	s.lossName = loss.Name()
	s.optimizer = optimizer

	currentCols := inputCols
	for i, l := range s.layers {
		var out int
		if il, ok := l.(layer.ImageLayer); ok {
			if img == (layer.Image{}) {
				return fmt.Errorf("tensai: layer %d init: %T needs a spatial input shape; compile the model with CompileImage", i, l)
			}
			outImg, err := il.InitImage(img, s.rng)
			if err != nil {
				return fmt.Errorf("tensai: layer %d init: %w", i, err)
			}
			img = outImg
			out = outImg.Cols()
		} else {
			var err error
			out, err = l.Init(currentCols, s.rng)
			if err != nil {
				return fmt.Errorf("tensai: layer %d init: %w", i, err)
			}
			if out != currentCols {
				// The layer reshaped the row; the spatial reading is gone.
				img = layer.Image{}
			}
		}
		// Layers that expose parameters register with the optimizer.
		if w, _ := l.Params(); w != nil {
			_ = optimizer.NewLayer()
		}
		if out <= 0 {
			return fmt.Errorf("tensai: layer %d produced non-positive output cols: %d", i, out)
		}
		currentCols = out
	}
	return nil
}

// forward runs the full stack and returns the output of the last layer.
func (s *Sequential) forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	current := input
	for i, l := range s.layers {
		out, err := l.Forward(current)
		if err != nil {
			return nil, fmt.Errorf("tensai: layer %d forward: %w", i, err)
		}
		current = out
	}
	return current, nil
}

// backward runs gradients back through the stack.
func (s *Sequential) backward(grad *tensai.Matrix) error {
	current := grad
	for i := len(s.layers) - 1; i >= 0; i-- {
		out, err := s.layers[i].Backward(current)
		if err != nil {
			return fmt.Errorf("tensai: layer %d backward: %w", i, err)
		}
		current = out
	}
	return nil
}

// applyGrads updates each parameterized layer via the optimizer.
func (s *Sequential) applyGrads() {
	idx := 0
	for _, l := range s.layers {
		weights, bias := l.Params()
		if weights == nil {
			continue
		}
		gradW, gradB := l.Grads()
		s.optimizer.Step(idx, weights, gradW, bias, gradB)
		idx++
	}
}

// trainable is implemented by layers that behave differently during training
// and inference (e.g. Dropout, BatchNorm).
type trainable interface {
	SetTraining(bool)
}

// setTraining switches all mode-aware layers between train and eval.
func (s *Sequential) setTraining(train bool) {
	for _, l := range s.layers {
		if t, ok := l.(trainable); ok {
			t.SetTraining(train)
		}
	}
}

// FitStep performs one forward + loss + backward + update pass for a batch
// and returns the average loss for that batch.
func (s *Sequential) FitStep(input, target *tensai.Matrix) (tensai.Float, error) {
	s.setTraining(true)
	defer s.setTraining(false)
	pred, err := s.forward(input)
	if err != nil {
		return 0, err
	}
	var lossVal tensai.Float
	var grad *tensai.Matrix
	if li, ok := s.loss.(lossInto); ok {
		s.lossGrad = tensai.EnsureMatrix(s.lossGrad, pred.Rows, pred.Cols)
		lossVal, err = li.LossInto(pred, target, s.lossGrad)
		grad = s.lossGrad
		if err != nil {
			return 0, fmt.Errorf("tensai: loss: %w", err)
		}
	} else {
		lossVal, grad, err = s.loss.Loss(pred, target)
		if err != nil {
			return 0, fmt.Errorf("tensai: loss: %w", err)
		}
	}
	if err := s.backward(grad); err != nil {
		return 0, err
	}
	s.applyGrads()
	return lossVal, nil
}

// Fit trains the model for the given number of epochs over the dataset.
// If epochs > 1 the full dataset is reused each epoch (full-batch by default;
// callers can pass minibatches to FitStep directly for finer control).
func (s *Sequential) Fit(input, target *tensai.Matrix, epochs int) error {
	if input.Rows != target.Rows {
		return fmt.Errorf("tensai: fit row mismatch: inputs=%d targets=%d", input.Rows, target.Rows)
	}
	fmt.Printf("tensai: training %d epochs with %s + %s on %d samples\n",
		epochs, s.optimizer.Name(), s.lossName, input.Rows)
	for e := 1; e <= epochs; e++ {
		lossVal, err := s.FitStep(input, target)
		if err != nil {
			return err
		}
		if e == 1 || e%500 == 0 || e == epochs {
			fmt.Printf("  epoch %5d: loss=%.6f\n", e, lossVal)
		}
	}
	return nil
}

// Predict runs a forward pass with no gradient tracking.
func (s *Sequential) Predict(input *tensai.Matrix) (*tensai.Matrix, error) {
	return s.forward(input)
}

// Layers returns the layers in forward order, for tools that walk the
// model structure (e.g. format exporters).
func (s *Sequential) Layers() []layer.Layer {
	return s.layers
}
