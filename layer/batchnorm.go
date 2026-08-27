package layer

import (
	"fmt"
	"math/rand"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// BatchNorm normalizes each feature column over the batch, then applies a
// learned scale (gamma) and shift (beta). During training it normalizes with
// batch statistics and maintains running estimates; during inference it uses
// the running estimates.
//
// gamma is exposed as the layer's weights (a 1xC matrix) and beta as its
// bias, so optimizers update them like any other parameters.
type BatchNorm struct {
	Momentum tensai.Float // running-stats decay, default 0.9
	Eps      tensai.Float // numerical stability, default 1e-5

	gamma *tensai.Matrix // 1 x C
	beta  []tensai.Float

	gradGamma *tensai.Matrix
	gradBeta  []tensai.Float

	runMean []tensai.Float
	runVar  []tensai.Float

	xhat   *tensai.Matrix // cached normalized input from the last training forward
	invStd []tensai.Float

	bufferPair
}

// NewBatchNorm returns a BatchNorm layer with standard defaults.
func NewBatchNorm() *BatchNorm {
	return &BatchNorm{Momentum: 0.9, Eps: 1e-5}
}

func (b *BatchNorm) Init(inputCols int, _ *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: batchnorm init with non-positive input cols: %d", inputCols)
	}
	b.gamma = tensai.NewMatrix(1, inputCols)
	for i := range b.gamma.Data {
		b.gamma.Data[i] = 1
	}
	b.beta = make([]tensai.Float, inputCols)
	b.gradGamma = tensai.NewMatrix(1, inputCols)
	b.gradBeta = make([]tensai.Float, inputCols)
	b.runMean = make([]tensai.Float, inputCols)
	b.runVar = make([]tensai.Float, inputCols)
	for i := range b.runVar {
		b.runVar[i] = 1
	}
	return inputCols, nil
}

func (b *BatchNorm) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Cols != b.gamma.Cols {
		return nil, fmt.Errorf("tensai: batchnorm forward shape mismatch: input cols=%d, layer cols=%d",
			input.Cols, b.gamma.Cols)
	}
	cols := input.Cols
	out := b.fwdBuf(input.Rows, cols)

	if !b.training {
		b.xhat = nil
		for r := 0; r < input.Rows; r++ {
			for c := 0; c < cols; c++ {
				x := input.Data[r*cols+c]
				xn := (x - b.runMean[c]) / kernels.SqrtF(b.runVar[c]+b.Eps)
				out.Data[r*cols+c] = b.gamma.Data[c]*xn + b.beta[c]
			}
		}
		return out, nil
	}

	m := tensai.Float(input.Rows)
	b.xhat = tensai.EnsureMatrix(b.xhat, input.Rows, cols)
	if cap(b.invStd) < cols {
		b.invStd = make([]tensai.Float, cols)
	} else {
		b.invStd = b.invStd[:cols]
	}
	for c := 0; c < cols; c++ {
		var mean tensai.Float
		for r := 0; r < input.Rows; r++ {
			mean += input.Data[r*cols+c]
		}
		mean /= m
		var variance tensai.Float
		for r := 0; r < input.Rows; r++ {
			d := input.Data[r*cols+c] - mean
			variance += d * d
		}
		variance /= m
		b.invStd[c] = 1 / kernels.SqrtF(variance+b.Eps)
		for r := 0; r < input.Rows; r++ {
			xn := (input.Data[r*cols+c] - mean) * b.invStd[c]
			b.xhat.Data[r*cols+c] = xn
			out.Data[r*cols+c] = b.gamma.Data[c]*xn + b.beta[c]
		}
		b.runMean[c] = b.Momentum*b.runMean[c] + (1-b.Momentum)*mean
		b.runVar[c] = b.Momentum*b.runVar[c] + (1-b.Momentum)*variance
	}
	return out, nil
}

func (b *BatchNorm) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if b.xhat == nil {
		return nil, fmt.Errorf("tensai: batchnorm backward called before a training forward")
	}
	if gradOutput.Rows != b.xhat.Rows || gradOutput.Cols != b.xhat.Cols {
		return nil, fmt.Errorf("tensai: batchnorm backward shape mismatch: grad %dx%d, cached %dx%d",
			gradOutput.Rows, gradOutput.Cols, b.xhat.Rows, b.xhat.Cols)
	}
	cols := gradOutput.Cols
	m := tensai.Float(gradOutput.Rows)
	out := b.bwdBuf(gradOutput.Rows, cols)
	for c := 0; c < cols; c++ {
		var sumG, sumGX tensai.Float
		for r := 0; r < gradOutput.Rows; r++ {
			g := gradOutput.Data[r*cols+c]
			sumG += g
			sumGX += g * b.xhat.Data[r*cols+c]
		}
		b.gradGamma.Data[c] = sumGX
		b.gradBeta[c] = sumG
		// dL/dx accounting for x's effect on the batch mean and variance:
		// dx = gamma*invStd/m * (m*g - sum(g) - xhat*sum(g*xhat))
		k := b.gamma.Data[c] * b.invStd[c] / m
		for r := 0; r < gradOutput.Rows; r++ {
			g := gradOutput.Data[r*cols+c]
			out.Data[r*cols+c] = k * (m*g - sumG - b.xhat.Data[r*cols+c]*sumGX)
		}
	}
	return out, nil
}

func (b *BatchNorm) Params() (*tensai.Matrix, []tensai.Float) { return b.gamma, b.beta }
func (b *BatchNorm) Grads() (*tensai.Matrix, []tensai.Float)  { return b.gradGamma, b.gradBeta }

func (b *BatchNorm) SetParams(weights *tensai.Matrix, bias []tensai.Float) error {
	if weights == nil || weights.Rows != 1 || len(bias) != weights.Cols {
		return fmt.Errorf("tensai: batchnorm SetParams mismatch")
	}
	b.gamma = weights
	b.beta = bias
	return nil
}

// ExtraState exposes the running statistics for serialization.
func (b *BatchNorm) ExtraState() map[string][]tensai.Float {
	return map[string][]tensai.Float{
		"running_mean": b.runMean,
		"running_var":  b.runVar,
	}
}

func (b *BatchNorm) SetExtraState(state map[string][]tensai.Float) error {
	mean, okM := state["running_mean"]
	variance, okV := state["running_var"]
	if !okM || !okV || len(mean) != b.gamma.Cols || len(variance) != b.gamma.Cols {
		return fmt.Errorf("tensai: batchnorm state mismatch")
	}
	b.runMean = mean
	b.runVar = variance
	return nil
}

// RunningStats returns the running mean and variance estimates used at
// inference time, for exporters.
func (b *BatchNorm) RunningStats() (mean, variance []tensai.Float) {
	return b.runMean, b.runVar
}
