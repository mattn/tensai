package main

import (
	"fmt"
	"math"
	"math/rand"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/metrics"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const (
	numClasses     = 3
	featuresPerRow = 2
	pointsPerClass = 120
	testFraction   = 0.25
	hiddenWidth    = 64
	epochs         = 700
	batchSize      = 64
	turns          = 3.5
	noise          = 0.18
	trainingSeed   = 1
)

func makeSpiral() *dataset.Dataset {
	rng := rand.New(rand.NewSource(trainingSeed))
	inputs := tensai.NewMatrix(numClasses*pointsPerClass, featuresPerRow)
	targets := tensai.NewMatrix(numClasses*pointsPerClass, 1)
	for cls := 0; cls < numClasses; cls++ {
		for i := 0; i < pointsPerClass; i++ {
			r := float64(i) / float64(pointsPerClass-1)
			theta := float64(cls)*2*math.Pi/numClasses + turns*r + rng.NormFloat64()*noise
			row := cls*pointsPerClass + i
			inputs.Set(row, 0, float32(r*math.Cos(theta)))
			inputs.Set(row, 1, float32(r*math.Sin(theta)))
			targets.Set(row, 0, float32(cls))
		}
	}
	ds, err := dataset.New(inputs, targets)
	if err != nil {
		panic(err)
	}
	return ds
}

func evaluate(model *model.Sequential, ds *dataset.Dataset) (int, [][]int, error) {
	pred, err := model.Predict(ds.Inputs)
	if err != nil {
		return 0, nil, err
	}
	correct, err := metrics.Correct(pred, ds.Targets)
	if err != nil {
		return 0, nil, err
	}
	confusion, err := metrics.Confusion(pred, ds.Targets)
	if err != nil {
		return 0, nil, err
	}
	return correct, confusion, nil
}

func main() {
	splitRng := rand.New(rand.NewSource(trainingSeed + 2))
	train, test, err := makeSpiral().SplitStratified(testFraction, splitRng)
	if err != nil {
		panic(err)
	}

	model := model.NewSequential()
	model.Add(layer.NewDense(hiddenWidth))
	model.Add(&layer.Tanh{})
	model.Add(layer.NewDense(hiddenWidth))
	model.Add(&layer.Tanh{})
	model.Add(layer.NewDense(numClasses))
	if err := model.Compile(featuresPerRow, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.01)); err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(trainingSeed + 1))
	for epoch := 1; epoch <= epochs; epoch++ {
		var lossSum float32
		var steps int
		err := train.Batches(batchSize, rng, func(in, tgt *tensai.Matrix) error {
			lossVal, err := model.FitStep(in, tgt)
			lossSum += lossVal
			steps++
			return err
		})
		if err != nil {
			panic(err)
		}
		if epoch == 1 || epoch%100 == 0 || epoch == epochs {
			fmt.Printf("epoch %4d: loss=%.6f\n", epoch, lossSum/float32(steps))
		}
	}

	trainCorrect, _, err := evaluate(model, train)
	if err != nil {
		panic(err)
	}
	testCorrect, confusion, err := evaluate(model, test)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\ntrain accuracy: %d/%d\n", trainCorrect, train.Len())
	fmt.Printf("test accuracy:  %d/%d\n", testCorrect, test.Len())
	fmt.Println("\nconfusion matrix (rows = actual, columns = predicted):")
	for r := range confusion {
		fmt.Printf("  class %d:", r)
		for _, n := range confusion[r] {
			fmt.Printf(" %3d", n)
		}
		fmt.Println()
	}
}
