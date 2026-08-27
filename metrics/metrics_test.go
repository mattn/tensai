package metrics

import (
	"testing"

	"github.com/mattn/tensai"
)

func fixtures(t *testing.T) (*tensai.Matrix, *tensai.Matrix) {
	t.Helper()
	// Rows 0-3 predict classes 1, 0, 2, 2; targets are 1, 0, 2, 1.
	pred, err := tensai.NewMatrixFromSlice(4, 3, []tensai.Float{
		0.1, 0.8, 0.1,
		0.9, 0.05, 0.05,
		0.2, 0.2, 0.6,
		0.1, 0.3, 0.6,
	})
	if err != nil {
		t.Fatal(err)
	}
	targets, err := tensai.NewMatrixFromSlice(4, 1, []tensai.Float{1, 0, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	return pred, targets
}

func TestCorrectAndAccuracy(t *testing.T) {
	pred, targets := fixtures(t)
	correct, err := Correct(pred, targets)
	if err != nil {
		t.Fatal(err)
	}
	if correct != 3 {
		t.Fatalf("Correct = %d, want 3", correct)
	}
	acc, err := Accuracy(pred, targets)
	if err != nil {
		t.Fatal(err)
	}
	if acc != 0.75 {
		t.Fatalf("Accuracy = %g, want 0.75", acc)
	}
}

func TestConfusion(t *testing.T) {
	pred, targets := fixtures(t)
	confusion, err := Confusion(pred, targets)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]int{
		{1, 0, 0},
		{0, 1, 1},
		{0, 0, 1},
	}
	for r := range want {
		for c := range want[r] {
			if confusion[r][c] != want[r][c] {
				t.Fatalf("confusion[%d][%d] = %d, want %d", r, c, confusion[r][c], want[r][c])
			}
		}
	}
}

func TestValidation(t *testing.T) {
	pred, targets := fixtures(t)
	if _, err := Correct(nil, targets); err == nil {
		t.Fatal("nil pred did not error")
	}
	short, err := tensai.NewMatrixFromSlice(2, 1, []tensai.Float{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Accuracy(pred, short); err == nil {
		t.Fatal("row mismatch did not error")
	}
	bad, err := tensai.NewMatrixFromSlice(4, 1, []tensai.Float{1, 0, 2, 5})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Confusion(pred, bad); err == nil {
		t.Fatal("out-of-range target class did not error")
	}
}
