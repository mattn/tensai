package dataset

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

// makeClassDataset builds n samples whose target cycles through classes
// and whose single input column encodes the original row index.
func makeClassDataset(t *testing.T, n, classes int) *Dataset {
	t.Helper()
	in := tensai.NewMatrix(n, 1)
	tgt := tensai.NewMatrix(n, 1)
	for i := 0; i < n; i++ {
		in.Set(i, 0, tensai.Float(i))
		tgt.Set(i, 0, tensai.Float(i%classes))
	}
	ds, err := New(in, tgt)
	if err != nil {
		t.Fatal(err)
	}
	return ds
}

func TestSubsetCopiesRows(t *testing.T) {
	ds := makeClassDataset(t, 10, 2)
	sub, err := ds.Subset([]int{7, 0, 3})
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range []int{7, 0, 3} {
		if got := sub.Inputs.At(i, 0); got != tensai.Float(r) {
			t.Fatalf("row %d input = %g, want %d", i, got, r)
		}
		if got := sub.Targets.At(i, 0); got != tensai.Float(r%2) {
			t.Fatalf("row %d target = %g, want %d", i, got, r%2)
		}
	}
	// Copies, not views: mutating the subset must not touch the original.
	sub.Inputs.Set(0, 0, -1)
	if ds.Inputs.At(7, 0) != 7 {
		t.Fatal("subset shares memory with the original dataset")
	}
}

func TestSubsetRejectsBadRows(t *testing.T) {
	ds := makeClassDataset(t, 5, 2)
	if _, err := ds.Subset(nil); err == nil {
		t.Fatal("empty subset did not error")
	}
	if _, err := ds.Subset([]int{4, 5}); err == nil {
		t.Fatal("out-of-range row did not error")
	}
}

func TestSplitStratifiedKeepsClassBalance(t *testing.T) {
	const n, classes = 150, 3
	ds := makeClassDataset(t, n, classes)
	train, test, err := ds.SplitStratified(0.2, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}
	if train.Len() != 120 || test.Len() != 30 {
		t.Fatalf("train=%d test=%d, want 120 and 30", train.Len(), test.Len())
	}
	count := func(ds *Dataset) map[int]int {
		m := map[int]int{}
		for r := 0; r < ds.Len(); r++ {
			m[int(ds.Targets.At(r, 0))]++
		}
		return m
	}
	for cls, got := range count(train) {
		if got != 40 {
			t.Fatalf("train class %d has %d rows, want 40", cls, got)
		}
	}
	for cls, got := range count(test) {
		if got != 10 {
			t.Fatalf("test class %d has %d rows, want 10", cls, got)
		}
	}
	// Every original row appears exactly once across the two halves.
	seen := map[int]bool{}
	for _, half := range []*Dataset{train, test} {
		for r := 0; r < half.Len(); r++ {
			id := int(half.Inputs.At(r, 0))
			if seen[id] {
				t.Fatalf("row %d appears twice", id)
			}
			seen[id] = true
		}
	}
	if len(seen) != n {
		t.Fatalf("%d unique rows after split, want %d", len(seen), n)
	}
}

func TestSplitStratifiedNilRNGKeepsOrder(t *testing.T) {
	ds := makeClassDataset(t, 10, 2)
	train, _, err := ds.SplitStratified(0.2, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Class 0 rows (0,2,4,6,8) minus one test row: the first four stay in
	// original order at the head of the training split.
	for i, want := range []tensai.Float{0, 2, 4, 6} {
		if got := train.Inputs.At(i, 0); got != want {
			t.Fatalf("train row %d = %g, want %g", i, got, want)
		}
	}
}

func TestSplitStratifiedRejectsBadInput(t *testing.T) {
	ds := makeClassDataset(t, 10, 2)
	if _, _, err := ds.SplitStratified(0, nil); err == nil {
		t.Fatal("fraction 0 did not error")
	}
	if _, _, err := ds.SplitStratified(1, nil); err == nil {
		t.Fatal("fraction 1 did not error")
	}
	wide := &Dataset{Inputs: tensai.NewMatrix(4, 1), Targets: tensai.NewMatrix(4, 2)}
	if _, _, err := wide.SplitStratified(0.5, nil); err == nil {
		t.Fatal("multi-column targets did not error")
	}
}
