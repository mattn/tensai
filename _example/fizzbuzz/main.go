package main

import (
	"fmt"
	"math/rand"

	tensai "github.com/mattn/tensai"
)

const (
	numBits     = 10 // binary encoding covers 1..1023
	numFeatures = numBits + 3 + 5
	numClasses  = 4 // number, Fizz, Buzz, FizzBuzz
)

// encode returns binary features plus explicit modulo cycles for 3 and 5.
func encode(n int) []float32 {
	features := make([]float32, numFeatures)
	for i := 0; i < numBits; i++ {
		features[i] = float32(n >> i & 1)
	}
	features[numBits+n%3] = 1
	features[numBits+3+n%5] = 1
	return features
}

// class returns the FizzBuzz class of n.
func class(n int) int {
	switch {
	case n%15 == 0:
		return 3
	case n%5 == 0:
		return 2
	case n%3 == 0:
		return 1
	default:
		return 0
	}
}

// label renders n according to its class.
func label(n, cls int) string {
	switch cls {
	case 3:
		return "FizzBuzz"
	case 2:
		return "Buzz"
	case 1:
		return "Fizz"
	default:
		return fmt.Sprint(n)
	}
}

// dataset builds input/target matrices for the numbers lo..hi inclusive.
func dataset(lo, hi int) (*tensai.Matrix, *tensai.Matrix, error) {
	rows := hi - lo + 1
	inputData := make([]float32, 0, rows*numFeatures)
	targetData := make([]float32, 0, rows)
	for n := lo; n <= hi; n++ {
		inputData = append(inputData, encode(n)...)
		targetData = append(targetData, float32(class(n)))
	}
	inputs, err := tensai.NewMatrixFromSlice(rows, numFeatures, inputData)
	if err != nil {
		return nil, nil, err
	}
	targets, err := tensai.NewMatrixFromSlice(rows, 1, targetData)
	if err != nil {
		return nil, nil, err
	}
	return inputs, targets, nil
}

func main() {
	// The classic "FizzBuzz as machine learning" joke: train on 101..1023
	// and see whether the network generalizes to the unseen 1..100.
	model := tensai.NewSequential()
	model.Add(tensai.NewDense(16))
	model.Add(&tensai.ReLU{})
	model.Add(tensai.NewDense(numClasses))

	adam := tensai.NewAdam(0.02)
	if err := model.Compile(numFeatures, tensai.SoftmaxCrossEntropy{}, adam); err != nil {
		panic(err)
	}

	trainIn, trainTgt, err := dataset(101, 1<<numBits-1)
	if err != nil {
		panic(err)
	}

	// Shuffled mini-batches converge far faster than full-batch epochs, so we
	// drive FitStep directly instead of using model.Fit.
	const (
		epochs    = 40
		batchSize = 128
	)
	rng := rand.New(rand.NewSource(1))
	batchIn := tensai.NewMatrix(batchSize, numFeatures)
	batchTgt := tensai.NewMatrix(batchSize, 1)
	for epoch := 1; epoch <= epochs; epoch++ {
		perm := rng.Perm(trainIn.Rows)
		var lossSum float32
		var steps int
		for off := 0; off+batchSize <= len(perm); off += batchSize {
			for i, r := range perm[off : off+batchSize] {
				copy(batchIn.Data[i*numFeatures:(i+1)*numFeatures], trainIn.Data[r*numFeatures:(r+1)*numFeatures])
				batchTgt.Data[i] = trainTgt.Data[r]
			}
			lossVal, err := model.FitStep(batchIn, batchTgt)
			if err != nil {
				panic(err)
			}
			lossSum += lossVal
			steps++
		}
		if epoch == 1 || epoch%100 == 0 || epoch == epochs {
			fmt.Printf("epoch %4d: loss=%.6f\n", epoch, lossSum/float32(steps))
		}
	}

	testIn, testTgt, err := dataset(1, 100)
	if err != nil {
		panic(err)
	}
	pred, err := model.Predict(testIn)
	if err != nil {
		panic(err)
	}

	fmt.Println("\nFizzBuzz 1..100 from the trained network:")
	correct := 0
	for r := 0; r < pred.Rows; r++ {
		best := 0
		for c := 1; c < pred.Cols; c++ {
			if pred.At(r, c) > pred.At(r, best) {
				best = c
			}
		}
		n := r + 1
		want := int(testTgt.At(r, 0))
		mark := ""
		if best == want {
			correct++
		} else {
			mark = fmt.Sprintf("   (wrong, want %s)", label(n, want))
		}
		fmt.Printf("%4d: %-8s%s\n", n, label(n, best), mark)
	}
	fmt.Printf("\naccuracy: %d/100\n", correct)
}
