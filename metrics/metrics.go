// Package metrics provides evaluation helpers for classification models.
//
// Predictions are matrices with one column of scores per class, as the
// classification losses produce (raw logits are fine — only the row-wise
// argmax matters). Targets are a single column of class indices.
package metrics

import (
	"fmt"

	"github.com/mattn/tensai"
)

func check(pred, targets *tensai.Matrix) error {
	if pred == nil || targets == nil {
		return fmt.Errorf("tensai: metrics: nil matrix")
	}
	if targets.Cols != 1 {
		return fmt.Errorf("tensai: metrics: targets must have one column, got %d", targets.Cols)
	}
	if pred.Rows != targets.Rows {
		return fmt.Errorf("tensai: metrics: row mismatch: pred=%d targets=%d", pred.Rows, targets.Rows)
	}
	if pred.Rows == 0 || pred.Cols == 0 {
		return fmt.Errorf("tensai: metrics: empty predictions")
	}
	return nil
}

// Correct returns how many rows of pred have their argmax equal to the
// class index in targets.
func Correct(pred, targets *tensai.Matrix) (int, error) {
	if err := check(pred, targets); err != nil {
		return 0, err
	}
	correct := 0
	for r := 0; r < pred.Rows; r++ {
		if pred.ArgmaxRow(r) == int(targets.At(r, 0)) {
			correct++
		}
	}
	return correct, nil
}

// Accuracy returns the fraction of rows whose argmax prediction matches
// the target class.
func Accuracy(pred, targets *tensai.Matrix) (float64, error) {
	correct, err := Correct(pred, targets)
	if err != nil {
		return 0, err
	}
	return float64(correct) / float64(pred.Rows), nil
}

// Confusion returns the confusion matrix: rows are actual classes,
// columns are predicted classes. The matrix has one row and column per
// prediction column, so every class appears even when unseen.
func Confusion(pred, targets *tensai.Matrix) ([][]int, error) {
	if err := check(pred, targets); err != nil {
		return nil, err
	}
	n := pred.Cols
	confusion := make([][]int, n)
	for i := range confusion {
		confusion[i] = make([]int, n)
	}
	for r := 0; r < pred.Rows; r++ {
		want := int(targets.At(r, 0))
		if want < 0 || want >= n {
			return nil, fmt.Errorf("tensai: metrics: target class %d out of range [0,%d)", want, n)
		}
		confusion[want][pred.ArgmaxRow(r)]++
	}
	return confusion, nil
}
