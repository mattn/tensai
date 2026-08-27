package optim

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// Optimizer is an update-rule configuration (learning rate, momentum, ...).
// New hands out an Updater with fresh state for one parameter pair; the
// model keeps one Updater per parameterized layer, so there is no index
// bookkeeping between the model and the optimizer.
type Optimizer interface {
	// New returns an Updater holding fresh per-parameter state.
	New() Updater
	// Name returns a short identifier.
	Name() string
}

// Updater applies the optimizer rule to one (weights, bias) parameter pair,
// carrying that pair's state (momentum buffers, Adam moments, step count).
type Updater interface {
	Step(weights, gradW *tensai.Matrix, bias, gradB []tensai.Float)
}

// SGD is stochastic gradient descent with optional momentum.
type SGD struct {
	LR       tensai.Float
	Momentum tensai.Float
}

// NewSGD returns an SGD optimizer. Momentum of 0 disables momentum.
func NewSGD(lr, momentum tensai.Float) *SGD {
	return &SGD{LR: lr, Momentum: momentum}
}

// Name returns "sgd".
func (s *SGD) Name() string { return "sgd" }

// New returns an updater with fresh velocity buffers.
func (s *SGD) New() Updater { return &sgdUpdater{cfg: s} }

type sgdUpdater struct {
	cfg  *SGD
	velW *tensai.Matrix
	velB []tensai.Float
}

// Step updates parameters with standard (momentum) SGD.
func (u *sgdUpdater) Step(weights, gradW *tensai.Matrix, bias, gradB []tensai.Float) {
	if u.velW == nil {
		u.velW = tensai.NewMatrix(weights.Rows, weights.Cols)
		u.velB = make([]tensai.Float, len(bias))
	}
	kernels.SGDStep(weights.Data, gradW.Data, u.velW.Data, u.cfg.Momentum, u.cfg.LR)
	kernels.SGDStep(bias, gradB, u.velB, u.cfg.Momentum, u.cfg.LR)
}

// Adam optimizer. With WeightDecay > 0 it becomes AdamW: decay is decoupled
// from the gradient update and applied to weights only (never to biases).
type Adam struct {
	LR          tensai.Float
	Beta1       tensai.Float
	Beta2       tensai.Float
	Eps         tensai.Float
	WeightDecay tensai.Float
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

// New returns an updater with fresh moment buffers and step count.
func (a *Adam) New() Updater { return &adamUpdater{cfg: a} }

type adamUpdater struct {
	cfg *Adam
	t   int
	mW  *tensai.Matrix
	vW  *tensai.Matrix
	mB  []tensai.Float
	vB  []tensai.Float
}

// Step updates parameters with the Adam rule. The bias-correction step
// count is per parameter, so a parameter that skips a step (an unused
// autograd node) keeps its correction exact.
func (u *adamUpdater) Step(weights, gradW *tensai.Matrix, bias, gradB []tensai.Float) {
	if u.mW == nil {
		u.mW = tensai.NewMatrix(weights.Rows, weights.Cols)
		u.vW = tensai.NewMatrix(weights.Rows, weights.Cols)
		u.mB = make([]tensai.Float, len(bias))
		u.vB = make([]tensai.Float, len(bias))
	}
	u.t++
	a := u.cfg

	// Bias corrections depend only on t, so hoist them out of the loops.
	rc1 := 1 / (1 - kernels.PowF(a.Beta1, tensai.Float(u.t)))
	rc2 := 1 / (1 - kernels.PowF(a.Beta2, tensai.Float(u.t)))

	kernels.AdamStep(weights.Data, gradW.Data, u.mW.Data, u.vW.Data,
		a.Beta1, a.Beta2, rc1, rc2, a.LR, a.Eps, a.WeightDecay)
	// Decoupled weight decay is never applied to biases.
	kernels.AdamStep(bias, gradB, u.mB, u.vB, a.Beta1, a.Beta2, rc1, rc2, a.LR, a.Eps, 0)
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
