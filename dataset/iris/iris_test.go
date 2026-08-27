package iris

import (
	"testing"

	"github.com/mattn/tensai"
)

func TestLoadShape(t *testing.T) {
	ds := Load()
	if ds.Inputs.Rows != SampleCount || ds.Inputs.Cols != FeatureCount {
		t.Fatalf("inputs shape = %dx%d, want %dx%d", ds.Inputs.Rows, ds.Inputs.Cols, SampleCount, FeatureCount)
	}
	if ds.Targets.Rows != SampleCount || ds.Targets.Cols != 1 {
		t.Fatalf("targets shape = %dx%d, want %dx1", ds.Targets.Rows, ds.Targets.Cols, SampleCount)
	}
	var perClass [ClassCount]int
	for r := 0; r < ds.Len(); r++ {
		cls := int(ds.Targets.At(r, 0))
		if cls < 0 || cls >= ClassCount {
			t.Fatalf("row %d: class %d out of range", r, cls)
		}
		perClass[cls]++
	}
	for cls, n := range perClass {
		if n != SampleCount/ClassCount {
			t.Fatalf("class %d has %d samples, want %d", cls, n, SampleCount/ClassCount)
		}
	}
}

func TestLoadKnownRows(t *testing.T) {
	ds := Load()
	first := []tensai.Float{5.1, 3.5, 1.4, 0.2}
	for c, want := range first {
		if got := ds.Inputs.At(0, c); got != want {
			t.Fatalf("row 0 col %d = %g, want %g", c, got, want)
		}
	}
	if got := ds.Targets.At(0, 0); got != 0 {
		t.Fatalf("row 0 class = %g, want 0", got)
	}
	last := []tensai.Float{5.9, 3.0, 5.1, 1.8}
	for c, want := range last {
		if got := ds.Inputs.At(SampleCount-1, c); got != want {
			t.Fatalf("last row col %d = %g, want %g", c, got, want)
		}
	}
	if got := ds.Targets.At(SampleCount-1, 0); got != 2 {
		t.Fatalf("last row class = %g, want 2", got)
	}
}

func TestLoadReturnsFreshCopy(t *testing.T) {
	a := Load()
	a.Inputs.Set(0, 0, -100)
	a.Targets.Set(0, 0, -1)
	b := Load()
	if b.Inputs.At(0, 0) == -100 || b.Targets.At(0, 0) == -1 {
		t.Fatal("mutating one Load result leaked into a later Load")
	}
}
