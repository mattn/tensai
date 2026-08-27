package knn

import (
	"math/rand"
	"testing"

	"github.com/mattn/tensai"
)

func TestKNNTwoClusters(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	const perClass = 50
	train := tensai.NewMatrix(2*perClass, 2)
	targets := tensai.NewMatrix(2*perClass, 1)
	for i := 0; i < perClass; i++ {
		train.Set(2*i, 0, tensai.Float(rng.NormFloat64())*0.5-2)
		train.Set(2*i, 1, tensai.Float(rng.NormFloat64())*0.5)
		targets.Data[2*i] = 0
		train.Set(2*i+1, 0, tensai.Float(rng.NormFloat64())*0.5+2)
		train.Set(2*i+1, 1, tensai.Float(rng.NormFloat64())*0.5)
		targets.Data[2*i+1] = 1
	}

	knn := New(5)
	if err := knn.Fit(train, targets); err != nil {
		t.Fatal(err)
	}
	test, err := tensai.NewMatrixFromSlice(2, 2, []tensai.Float{-2, 0, 2, 0})
	if err != nil {
		t.Fatal(err)
	}
	votes, err := knn.Predict(test)
	if err != nil {
		t.Fatal(err)
	}
	if votes.Rows != 2 || votes.Cols != 2 {
		t.Fatalf("expected 2x2 votes, got %dx%d", votes.Rows, votes.Cols)
	}
	if votes.At(0, 0) <= votes.At(0, 1) || votes.At(1, 1) <= votes.At(1, 0) {
		t.Fatalf("cluster centers misclassified: %v", votes.Data)
	}
	for r := 0; r < votes.Rows; r++ {
		if votes.At(r, 0)+votes.At(r, 1) != 5 {
			t.Fatalf("row %d votes do not sum to k: %v", r, votes.Data)
		}
	}
}

func TestKNNOneNeighborMemorizes(t *testing.T) {
	rng := rand.New(rand.NewSource(79))
	train := tensai.RandomMatrix(30, 4, rng)
	targets := tensai.NewMatrix(30, 1)
	for i := range targets.Data {
		targets.Data[i] = tensai.Float(i % 3)
	}
	knn := New(1)
	if err := knn.Fit(train, targets); err != nil {
		t.Fatal(err)
	}
	votes, err := knn.Predict(train)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < train.Rows; r++ {
		best := 0
		for c := 1; c < votes.Cols; c++ {
			if votes.At(r, c) > votes.At(r, best) {
				best = c
			}
		}
		if best != int(targets.Data[r]) {
			t.Fatalf("k=1 should memorize training point %d: got %d want %g",
				r, best, targets.Data[r])
		}
	}
}

func TestKNNValidation(t *testing.T) {
	knn := New(0)
	if err := knn.Fit(tensai.NewMatrix(3, 2), tensai.NewMatrix(3, 1)); err == nil {
		t.Error("k=0 should be rejected")
	}
	knn = New(2)
	if _, err := knn.Predict(tensai.NewMatrix(1, 2)); err == nil {
		t.Error("predict before fit should fail")
	}
	if err := knn.Fit(tensai.NewMatrix(3, 2), tensai.NewMatrix(2, 1)); err == nil {
		t.Error("target row mismatch should be rejected")
	}
	tgt := tensai.NewMatrix(3, 1)
	tgt.Data[1] = 1.5
	if err := knn.Fit(tensai.NewMatrix(3, 2), tgt); err == nil {
		t.Error("non-integer class labels should be rejected")
	}
}
