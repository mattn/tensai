package knn

import (
	"fmt"

	"github.com/mattn/tensai"
)

// Classifier is a k-nearest-neighbors classifier. It is a lazy learner: Fit just
// stores the training data, and Predict ranks neighbors by squared
// Euclidean distance. The distance computation is reduced to one matrix
// product per chunk (||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b), so it runs on
// the same tuned Dot kernel as the neural networks.
type Classifier struct {
	K int

	trainT  *tensai.Matrix // transposed training inputs (features x samples)
	sqNorm  []tensai.Float // squared norm per training sample
	labels  []int
	classes int
}

// New returns a classifier that votes among the k nearest neighbors.
func New(k int) *Classifier {
	return &Classifier{K: k}
}

// Fit stores the training set. Targets must be an Mx1 matrix of class
// indices, as with SoftmaxCrossEntropy.
func (k *Classifier) Fit(inputs, targets *tensai.Matrix) error {
	if err := inputs.Validate(); err != nil {
		return err
	}
	if k.K <= 0 || k.K > inputs.Rows {
		return fmt.Errorf("tensai: knn k=%d out of range [1,%d]", k.K, inputs.Rows)
	}
	if targets.Rows != inputs.Rows || targets.Cols != 1 {
		return fmt.Errorf("tensai: knn targets must be %dx1, got %dx%d",
			inputs.Rows, targets.Rows, targets.Cols)
	}
	k.trainT = inputs.T()
	k.sqNorm = make([]tensai.Float, inputs.Rows)
	k.labels = make([]int, inputs.Rows)
	k.classes = 0
	for r := 0; r < inputs.Rows; r++ {
		var sum tensai.Float
		row := inputs.Data[r*inputs.Cols : (r+1)*inputs.Cols]
		for _, v := range row {
			sum += v * v
		}
		k.sqNorm[r] = sum
		cls := int(targets.Data[r])
		if cls < 0 || tensai.Float(cls) != targets.Data[r] {
			return fmt.Errorf("tensai: knn target %d is not a class index: %g", r, targets.Data[r])
		}
		k.labels[r] = cls
		if cls+1 > k.classes {
			k.classes = cls + 1
		}
	}
	return nil
}

// Predict returns an (inputs.Rows x classes) matrix of neighbor vote counts;
// take the argmax per row for the predicted class. Row sums equal K.
func (k *Classifier) Predict(inputs *tensai.Matrix) (*tensai.Matrix, error) {
	if k.trainT == nil {
		return nil, fmt.Errorf("tensai: knn predict called before fit")
	}
	if err := inputs.Validate(); err != nil {
		return nil, err
	}
	if inputs.Cols != k.trainT.Rows {
		return nil, fmt.Errorf("tensai: knn feature mismatch: got %d, trained on %d",
			inputs.Cols, k.trainT.Rows)
	}
	train := k.trainT.Cols
	out := tensai.NewMatrix(inputs.Rows, k.classes)

	// Chunk the distance matrix to bound memory at ~chunk*train floats.
	chunk := 1 << 22 / train
	if chunk < 1 {
		chunk = 1
	}
	type neighbor struct {
		dist  tensai.Float
		label int
	}
	best := make([]neighbor, k.K)
	for off := 0; off < inputs.Rows; off += chunk {
		n := min(chunk, inputs.Rows-off)
		view := &tensai.Matrix{Rows: n, Cols: inputs.Cols, Data: inputs.Data[off*inputs.Cols : (off+n)*inputs.Cols]}
		prod, err := tensai.Dot(view, k.trainT) // n x train
		if err != nil {
			return nil, err
		}
		for r := 0; r < n; r++ {
			// Rank by ||b||^2 - 2*a.b: the test point's own norm is
			// constant per row and does not change the ordering.
			row := prod.Data[r*train : (r+1)*train]
			used := 0
			for c, dot := range row {
				d := k.sqNorm[c] - 2*dot
				if used == k.K && d >= best[used-1].dist {
					continue
				}
				i := used
				if used < k.K {
					used++
				} else {
					i = k.K - 1
				}
				for ; i > 0 && best[i-1].dist > d; i-- {
					best[i] = best[i-1]
				}
				best[i] = neighbor{dist: d, label: k.labels[c]}
			}
			for _, nb := range best[:used] {
				out.Data[(off+r)*k.classes+nb.label]++
			}
		}
	}
	return out, nil
}
