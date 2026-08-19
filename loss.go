package tensai

import (
	"errors"
	"fmt"
)

// Loss is a differentiable loss function operating on a prediction batch
// and a target batch of the same shape. It returns the scalar loss and the
// gradient of the loss with respect to the predictions.
type Loss interface {
	// Loss returns the average loss and the per-element gradient dL/dpred.
	Loss(pred, target *Matrix) (Float, *Matrix, error)
	// Name is a short identifier for logging.
	Name() string
}

// MeanSquaredError computes the average of squared differences.
type MeanSquaredError struct{}

// Name returns "mse".
func (MeanSquaredError) Name() string { return "mse" }

// Loss returns the mean squared error and its gradient.
func (MeanSquaredError) Loss(pred, target *Matrix) (Float, *Matrix, error) {
	if err := checkShapes(pred, target); err != nil {
		return 0, nil, err
	}
	n := Float(len(pred.Data))
	var sum Float
	grad := NewMatrix(pred.Rows, pred.Cols)
	for i := range pred.Data {
		diff := pred.Data[i] - target.Data[i]
		sum += diff * diff
		// d/dpred (1/n * (pred-target)^2) = 2/n * (pred-target)
		grad.Data[i] = 2.0 * diff / n
	}
	return sum / n, grad, nil
}

// SoftmaxCrossEntropy combines softmax + cross-entropy with integer class
// labels. Targets must be an Mx1 matrix whose entries are class indices.
type SoftmaxCrossEntropy struct{}

// Name returns "softmax_ce".
func (SoftmaxCrossEntropy) Name() string { return "softmax_ce" }

// Loss returns the average negative log-likelihood of the target classes
// and the combined softmax-cross-entropy gradient (pred - onehot) / batch.
func (SoftmaxCrossEntropy) Loss(pred, target *Matrix) (Float, *Matrix, error) {
	if pred.Rows != target.Rows {
		return 0, nil, fmt.Errorf("tensai: softmax_ce row mismatch: %d vs %d", pred.Rows, target.Rows)
	}
	if target.Cols != 1 {
		return 0, nil, fmt.Errorf("tensai: softmax_ce target must be Mx1, got %dx%d", target.Rows, target.Cols)
	}

	var loss Float
	grad := NewMatrix(pred.Rows, pred.Cols)
	invRows := 1 / Float(pred.Rows)
	for r := 0; r < pred.Rows; r++ {
		cls := int(target.Data[r])
		if cls < 0 || cls >= pred.Cols {
			return 0, nil, fmt.Errorf("tensai: softmax_ce class %d out of range [0,%d)", cls, pred.Cols)
		}

		// Numerically stable softmax for row r, with the probabilities
		// written straight into the gradient row.
		rowOff := r * pred.Cols
		row := pred.Data[rowOff : rowOff+pred.Cols]
		probs := grad.Data[rowOff : rowOff+pred.Cols]
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
		expShift(probs, row, maxVal)
		var denom Float
		for _, e := range probs {
			denom += e
		}
		scaleSlice(probs, 1/denom)

		loss -= logF(probs[cls] + 1e-12)

		// Gradient of softmax+CE is (p - onehot) / batch.
		scaleSlice(probs, invRows)
		probs[cls] -= invRows
	}
	return loss * invRows, grad, nil
}

// BinaryCrossEntropy computes the average binary cross-entropy between
// predicted probabilities in (0,1) and 0/1 targets of the same shape.
// Pair it with a Sigmoid output layer.
type BinaryCrossEntropy struct{}

// Name returns "bce".
func (BinaryCrossEntropy) Name() string { return "bce" }

// Loss returns the average binary cross-entropy and its gradient.
func (BinaryCrossEntropy) Loss(pred, target *Matrix) (Float, *Matrix, error) {
	if err := checkShapes(pred, target); err != nil {
		return 0, nil, err
	}
	const eps = 1e-12
	n := Float(len(pred.Data))
	var sum Float
	grad := NewMatrix(pred.Rows, pred.Cols)
	for i := range pred.Data {
		p := pred.Data[i]
		if p < eps {
			p = eps
		} else if p > 1-eps {
			p = 1 - eps
		}
		t := target.Data[i]
		sum -= t*logF(p) + (1-t)*logF(1-p)
		grad.Data[i] = (p - t) / (p * (1 - p) * n)
	}
	return sum / n, grad, nil
}

func checkShapes(a, b *Matrix) error {
	if a == nil || b == nil {
		return errors.New("tensai: nil matrix passed to loss")
	}
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return fmt.Errorf("tensai: shape mismatch: %dx%d vs %dx%d", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	return nil
}
