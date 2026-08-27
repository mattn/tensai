package optim

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// Optimizer updates a set of (weights, bias) parameter pairs using their
// gradients. One Optimizer instance is shared by the model; each
// parameterized layer gets its own state buffer inside the optimizer.
type Optimizer interface {
	// Step applies one update to the given parameters using their gradients.
	// It is called once per parameterized layer.
	Step(idx int, weights, gradW *tensai.Matrix, bias, gradB []tensai.Float)
	// NewLayer registers a new parameterized layer and returns its index.
	NewLayer() int
	// Name returns a short identifier.
	Name() string
}

// SGD is stochastic gradient descent with optional momentum.
type SGD struct {
	LR       tensai.Float
	Momentum tensai.Float
	velW     []*tensai.Matrix
	velB     [][]tensai.Float
}

// NewSGD returns an SGD optimizer. Momentum of 0 disables momentum.
func NewSGD(lr, momentum tensai.Float) *SGD {
	return &SGD{LR: lr, Momentum: momentum}
}

// Name returns "sgd".
func (s *SGD) Name() string { return "sgd" }

// NewLayer registers a layer and returns its index.
func (s *SGD) NewLayer() int {
	idx := len(s.velW)
	s.velW = append(s.velW, nil)
	s.velB = append(s.velB, nil)
	return idx
}

// Step updates parameters with standard (momentum) SGD.
func (s *SGD) Step(idx int, weights, gradW *tensai.Matrix, bias, gradB []tensai.Float) {
	if s.velW[idx] == nil {
		s.velW[idx] = tensai.NewMatrix(weights.Rows, weights.Cols)
	}
	if s.velB[idx] == nil {
		s.velB[idx] = make([]tensai.Float, len(bias))
	}
	kernels.SGDStep(weights.Data, gradW.Data, s.velW[idx].Data, s.Momentum, s.LR)
	kernels.SGDStep(bias, gradB, s.velB[idx], s.Momentum, s.LR)
}

// Adam optimizer. With WeightDecay > 0 it becomes AdamW: decay is decoupled
// from the gradient update and applied to weights only (never to biases).
type Adam struct {
	LR          tensai.Float
	Beta1       tensai.Float
	Beta2       tensai.Float
	Eps         tensai.Float
	WeightDecay tensai.Float
	t           int
	mW          []*tensai.Matrix
	vW          []*tensai.Matrix
	mB          [][]tensai.Float
	vB          [][]tensai.Float
}

// NewAdam returns an Adam optimizer with standard defaults.
func NewAdam(lr tensai.Float) *Adam {
	return &Adam{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
}

// NewAdamW returns an Adam optimizer with decoupled weight decay (AdamW).
func NewAdamW(lr, weightDecay tensai.Float) *Adam {
	a := NewAdam(lr)
	a.WeightDecay = weightDecay
	return a
}

// Name returns "adam".
func (a *Adam) Name() string { return "adam" }

// NewLayer registers a layer and returns its index.
func (a *Adam) NewLayer() int {
	idx := len(a.mW)
	a.mW = append(a.mW, nil)
	a.vW = append(a.vW, nil)
	a.mB = append(a.mB, nil)
	a.vB = append(a.vB, nil)
	return idx
}

// Step updates parameters with the Adam rule.
func (a *Adam) Step(idx int, weights, gradW *tensai.Matrix, bias, gradB []tensai.Float) {
	if a.mW[idx] == nil {
		a.mW[idx] = tensai.NewMatrix(weights.Rows, weights.Cols)
		a.vW[idx] = tensai.NewMatrix(weights.Rows, weights.Cols)
		a.mB[idx] = make([]tensai.Float, len(bias))
		a.vB[idx] = make([]tensai.Float, len(bias))
	}
	a.t++
	mW, vW := a.mW[idx], a.vW[idx]
	mB, vB := a.mB[idx], a.vB[idx]

	// Bias corrections depend only on t, so hoist them out of the loops.
	rc1 := 1 / (1 - kernels.PowF(a.Beta1, tensai.Float(a.t)))
	rc2 := 1 / (1 - kernels.PowF(a.Beta2, tensai.Float(a.t)))

	kernels.AdamStep(weights.Data, gradW.Data, mW.Data, vW.Data,
		a.Beta1, a.Beta2, rc1, rc2, a.LR, a.Eps, a.WeightDecay)
	// Decoupled weight decay is never applied to biases.
	kernels.AdamStep(bias, gradB, mB, vB, a.Beta1, a.Beta2, rc1, rc2, a.LR, a.Eps, 0)
}

// Assert Optimizer implementations conform to the interface at compile time.
var (
	_ Optimizer = (*SGD)(nil)
	_ Optimizer = (*Adam)(nil)
)

// String returns a human-readable description of an optimizer's config.
func (s *SGD) String() string {
	if s.Momentum > 0 {
		return fmt.Sprintf("sgd(lr=%g, momentum=%g)", s.LR, s.Momentum)
	}
	return fmt.Sprintf("sgd(lr=%g)", s.LR)
}

// String returns a human-readable description of an optimizer's config.
func (a *Adam) String() string {
	if a.WeightDecay > 0 {
		return fmt.Sprintf("adamw(lr=%g, b1=%g, b2=%g, wd=%g)", a.LR, a.Beta1, a.Beta2, a.WeightDecay)
	}
	return fmt.Sprintf("adam(lr=%g, b1=%g, b2=%g)", a.LR, a.Beta1, a.Beta2)
}
