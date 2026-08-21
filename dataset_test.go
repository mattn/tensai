package tensai

import (
	"math"
	"math/rand"
	"testing"
)

func makeDataset(t *testing.T, n int) *Dataset {
	t.Helper()
	in := NewMatrix(n, 3)
	tgt := NewMatrix(n, 1)
	for i := 0; i < n; i++ {
		for c := 0; c < 3; c++ {
			in.Set(i, c, Float(i*10+c))
		}
		tgt.Data[i] = Float(i)
	}
	ds, err := NewDataset(in, tgt)
	if err != nil {
		t.Fatal(err)
	}
	return ds
}

func TestDatasetShuffleKeepsRowsPaired(t *testing.T) {
	ds := makeDataset(t, 50)
	ds.Shuffle(rand.New(rand.NewSource(97)))
	seen := map[int]bool{}
	for i := 0; i < ds.Len(); i++ {
		id := int(ds.Targets.Data[i])
		if seen[id] {
			t.Fatalf("row %d duplicated after shuffle", id)
		}
		seen[id] = true
		for c := 0; c < 3; c++ {
			if ds.Inputs.At(i, c) != Float(id*10+c) {
				t.Fatalf("row %d no longer paired with its inputs", id)
			}
		}
	}
	if len(seen) != 50 {
		t.Fatalf("expected 50 unique rows, got %d", len(seen))
	}
}

func TestDatasetSplit(t *testing.T) {
	ds := makeDataset(t, 10)
	train, test, err := ds.Split(0.2)
	if err != nil {
		t.Fatal(err)
	}
	if train.Len() != 8 || test.Len() != 2 {
		t.Fatalf("expected 8/2 split, got %d/%d", train.Len(), test.Len())
	}
	// Views share the underlying data.
	if &train.Inputs.Data[0] != &ds.Inputs.Data[0] {
		t.Error("train view should share the parent's backing array")
	}
	if test.Targets.Data[0] != 8 {
		t.Errorf("test set should start at row 8, got %g", test.Targets.Data[0])
	}
	for _, bad := range []float64{0, 1, -0.5, 0.001} {
		if _, _, err := ds.Split(bad); err == nil {
			t.Errorf("fraction %g should be rejected", bad)
		}
	}
}

func TestDatasetBatches(t *testing.T) {
	ds := makeDataset(t, 10)
	var visited []int
	var lastBatch *Matrix
	err := ds.Batches(3, nil, func(in, tgt *Matrix) error {
		if in.Rows != 3 || tgt.Rows != 3 {
			t.Fatalf("batch shape %dx%d / %dx%d", in.Rows, in.Cols, tgt.Rows, tgt.Cols)
		}
		if lastBatch != nil && lastBatch != in {
			t.Fatal("batch buffer should be reused between calls")
		}
		lastBatch = in
		for i := 0; i < tgt.Rows; i++ {
			visited = append(visited, int(tgt.Data[i]))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// 10 samples at batch 3: 3 batches, trailing sample skipped.
	if len(visited) != 9 {
		t.Fatalf("expected 9 visited rows, got %d: %v", len(visited), visited)
	}
	for i, id := range visited {
		if id != i {
			t.Fatalf("nil rng should visit rows in order: %v", visited)
		}
	}

	// Shuffled epochs visit each sample at most once.
	seen := map[int]int{}
	if err := ds.Batches(5, rand.New(rand.NewSource(3)), func(in, tgt *Matrix) error {
		for i := 0; i < tgt.Rows; i++ {
			seen[int(tgt.Data[i])]++
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 10 {
		t.Fatalf("batch 5 over 10 samples should visit all rows once, saw %d", len(seen))
	}

	if err := ds.Batches(11, nil, func(_, _ *Matrix) error { return nil }); err == nil {
		t.Error("oversized batch should be rejected")
	}
}

func TestDatasetStandardize(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	in := NewMatrix(200, 2)
	for i := range in.Data {
		in.Data[i] = Float(rng.NormFloat64()*3 + 5)
	}
	ds, err := NewDataset(in, NewMatrix(200, 1))
	if err != nil {
		t.Fatal(err)
	}
	mean, std := ds.Standardize()
	if len(mean) != 2 || len(std) != 2 {
		t.Fatalf("stats length %d/%d", len(mean), len(std))
	}
	for c := 0; c < 2; c++ {
		var m, v float64
		for r := 0; r < ds.Len(); r++ {
			m += float64(ds.Inputs.At(r, c))
		}
		m /= float64(ds.Len())
		for r := 0; r < ds.Len(); r++ {
			d := float64(ds.Inputs.At(r, c)) - m
			v += d * d
		}
		v /= float64(ds.Len())
		if math.Abs(m) > 1e-5 || math.Abs(v-1) > 1e-4 {
			t.Fatalf("col %d: mean=%g var=%g after standardize", c, m, v)
		}
	}

	// The same transform maps the original mean to zero on other data.
	other := NewMatrix(1, 2)
	other.Data[0], other.Data[1] = mean[0], mean[1]
	ods, _ := NewDataset(other, NewMatrix(1, 1))
	ods.StandardizeWith(mean, std)
	if math.Abs(float64(ods.Inputs.Data[0])) > 1e-6 {
		t.Fatalf("StandardizeWith should map the mean to 0, got %g", ods.Inputs.Data[0])
	}
}
