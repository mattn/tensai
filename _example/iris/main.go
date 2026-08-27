// Command iris trains a small classifier on Fisher's iris dataset using
// the built-in dataset/iris loader.
package main

import (
	"fmt"
	"math/rand"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset"
	"github.com/mattn/tensai/dataset/iris"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/metrics"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const (
	testFraction = 0.2
	epochs       = 900
	batchSize    = 32
	seed         = 3
)

func evaluate(net *model.Sequential, ds *dataset.Dataset) (int, [][]int, error) {
	pred, err := net.Predict(ds.Inputs)
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
	ds, err := iris.Load(nil)
	if err != nil {
		panic(err)
	}
	rng := rand.New(rand.NewSource(seed))
	train, test, err := ds.SplitStratified(testFraction, rng)
	if err != nil {
		panic(err)
	}
	// Standardize with training statistics only.
	mean, std := train.Standardize()
	test.StandardizeWith(mean, std)

	net := model.NewSequential()
	net.Add(layer.NewDense(12))
	net.Add(&layer.Tanh{})
	net.Add(layer.NewDense(iris.ClassCount))
	if err := net.Compile(iris.FeatureCount, loss.SoftmaxCrossEntropy{}, optim.NewAdam(0.03)); err != nil {
		panic(err)
	}

	batchRng := rand.New(rand.NewSource(seed + 1))
	for epoch := 1; epoch <= epochs; epoch++ {
		var lossSum float32
		var steps int
		err := train.Batches(batchSize, batchRng, func(in, tgt *tensai.Matrix) error {
			loss, err := net.FitStep(in, tgt)
			lossSum += loss
			steps++
			return err
		})
		if err != nil {
			panic(err)
		}
		if epoch == 1 || epoch%150 == 0 || epoch == epochs {
			fmt.Printf("epoch %4d: loss=%.6f\n", epoch, lossSum/float32(steps))
		}
	}

	trainCorrect, _, err := evaluate(net, train)
	if err != nil {
		panic(err)
	}
	testCorrect, confusion, err := evaluate(net, test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("\ntrain accuracy: %d/%d\n", trainCorrect, train.Len())
	fmt.Printf("test accuracy:  %d/%d\n", testCorrect, test.Len())
	fmt.Println("\nconfusion matrix (rows = actual, columns = predicted):")
	for r := range confusion {
		fmt.Printf("  %-10s:", iris.ClassNames[r])
		for _, n := range confusion[r] {
			fmt.Printf(" %3d", n)
		}
		fmt.Println()
	}
}
