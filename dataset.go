package tensai

import (
	"fmt"
	"math/rand"
)

// Dataset pairs an input matrix with its target matrix, row-aligned, and
// provides the usual training-data plumbing: shuffling, train/test
// splitting, mini-batch iteration, and standardization.
type Dataset struct {
	Inputs  *Matrix
	Targets *Matrix
}

// NewDataset wraps inputs and targets after checking row alignment.
func NewDataset(inputs, targets *Matrix) (*Dataset, error) {
	if err := inputs.Validate(); err != nil {
		return nil, err
	}
	if err := targets.Validate(); err != nil {
		return nil, err
	}
	if inputs.Rows != targets.Rows {
		return nil, fmt.Errorf("tensai: dataset row mismatch: inputs=%d targets=%d",
			inputs.Rows, targets.Rows)
	}
	return &Dataset{Inputs: inputs, Targets: targets}, nil
}

// Len returns the number of samples.
func (d *Dataset) Len() int { return d.Inputs.Rows }

// Shuffle permutes the samples in place, keeping input and target rows
// paired.
func (d *Dataset) Shuffle(rng *rand.Rand) {
	inCols, tgtCols := d.Inputs.Cols, d.Targets.Cols
	tmpIn := make([]Float, inCols)
	tmpTgt := make([]Float, tgtCols)
	for i := d.Len() - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		if i == j {
			continue
		}
		swapRow(d.Inputs.Data, i, j, inCols, tmpIn)
		swapRow(d.Targets.Data, i, j, tgtCols, tmpTgt)
	}
}

func swapRow(data []Float, i, j, cols int, tmp []Float) {
	a := data[i*cols : (i+1)*cols]
	b := data[j*cols : (j+1)*cols]
	copy(tmp, a)
	copy(a, b)
	copy(b, tmp)
}

// Split divides the dataset into a training and a test set, with
// testFraction (0 < f < 1) of the samples going to the test set. The two
// halves are views sharing the underlying data — no rows are copied.
// Shuffle first when the data is ordered.
func (d *Dataset) Split(testFraction float64) (train, test *Dataset, err error) {
	if testFraction <= 0 || testFraction >= 1 {
		return nil, nil, fmt.Errorf("tensai: split fraction must be in (0,1): %g", testFraction)
	}
	n := d.Len()
	testRows := int(float64(n)*testFraction + 0.5)
	if testRows == 0 || testRows == n {
		return nil, nil, fmt.Errorf("tensai: split of %d samples at %g leaves an empty side", n, testFraction)
	}
	trainRows := n - testRows
	view := func(m *Matrix, lo, rows int) *Matrix {
		return &Matrix{Rows: rows, Cols: m.Cols, Data: m.Data[lo*m.Cols : (lo+rows)*m.Cols]}
	}
	train = &Dataset{Inputs: view(d.Inputs, 0, trainRows), Targets: view(d.Targets, 0, trainRows)}
	test = &Dataset{Inputs: view(d.Inputs, trainRows, testRows), Targets: view(d.Targets, trainRows, testRows)}
	return train, test, nil
}

// Batches invokes fn once per mini-batch of exactly size samples, copying
// rows into buffers that are reused between calls (do not retain them).
// With a non-nil rng the visit order is reshuffled; trailing samples that
// do not fill a batch are skipped, matching common epoch loops.
func (d *Dataset) Batches(size int, rng *rand.Rand, fn func(inputs, targets *Matrix) error) error {
	if size <= 0 || size > d.Len() {
		return fmt.Errorf("tensai: batch size %d out of range [1,%d]", size, d.Len())
	}
	var perm []int
	if rng != nil {
		perm = rng.Perm(d.Len())
	} else {
		perm = make([]int, d.Len())
		for i := range perm {
			perm[i] = i
		}
	}
	inCols, tgtCols := d.Inputs.Cols, d.Targets.Cols
	bi := NewMatrix(size, inCols)
	bt := NewMatrix(size, tgtCols)
	for off := 0; off+size <= len(perm); off += size {
		for i, r := range perm[off : off+size] {
			copy(bi.Data[i*inCols:(i+1)*inCols], d.Inputs.Data[r*inCols:(r+1)*inCols])
			copy(bt.Data[i*tgtCols:(i+1)*tgtCols], d.Targets.Data[r*tgtCols:(r+1)*tgtCols])
		}
		if err := fn(bi, bt); err != nil {
			return err
		}
	}
	return nil
}

// Standardize scales every input column to zero mean and unit variance in
// place and returns the per-column statistics, for applying the same
// transform to other data with StandardizeWith. Constant columns keep a
// standard deviation of 1 so they pass through unchanged.
func (d *Dataset) Standardize() (mean, std []Float) {
	cols := d.Inputs.Cols
	n := Float(d.Len())
	mean = make([]Float, cols)
	std = make([]Float, cols)
	for r := 0; r < d.Len(); r++ {
		addSlice(mean, d.Inputs.Data[r*cols:(r+1)*cols])
	}
	for c := range mean {
		mean[c] /= n
	}
	for r := 0; r < d.Len(); r++ {
		row := d.Inputs.Data[r*cols : (r+1)*cols]
		for c, v := range row {
			diff := v - mean[c]
			std[c] += diff * diff
		}
	}
	for c := range std {
		std[c] = sqrtF(std[c] / n)
		if std[c] == 0 {
			std[c] = 1
		}
	}
	d.StandardizeWith(mean, std)
	return mean, std
}

// StandardizeWith applies previously computed statistics to the inputs in
// place (e.g. training-set statistics to a test set).
func (d *Dataset) StandardizeWith(mean, std []Float) {
	cols := d.Inputs.Cols
	for r := 0; r < d.Len(); r++ {
		row := d.Inputs.Data[r*cols : (r+1)*cols]
		for c := range row {
			row[c] = (row[c] - mean[c]) / std[c]
		}
	}
}
