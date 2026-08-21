// Command dataset walks through the Dataset workflow end to end: build a
// dataset, shuffle it, split off a test set, standardize using training
// statistics only, train with mini-batches, and evaluate on the held-out
// split.
package main

import (
	"fmt"
	"math/rand"

	tensai "github.com/mattn/tensai"
)

const (
	samples   = 600
	features  = 3
	classes   = 2
	epochs    = 60
	batchSize = 32
	seed      = 42
)

// synthesize builds a two-class dataset whose features live on wildly
// different scales (~1, ~100, ~10000), which is what Standardize is for.
func synthesize(rng *rand.Rand) *tensai.Dataset {
	inputs := tensai.NewMatrix(samples, features)
	targets := tensai.NewMatrix(samples, 1)
	for i := 0; i < samples; i++ {
		class := i % classes
		shift := float32(class)
		inputs.Set(i, 0, float32(rng.NormFloat64())+shift*1.5)
		inputs.Set(i, 1, float32(rng.NormFloat64())*80+300+shift*100)
		inputs.Set(i, 2, float32(rng.NormFloat64())*5000+20000-shift*6000)
		targets.Set(i, 0, float32(class))
	}
	ds, err := tensai.NewDataset(inputs, targets)
	if err != nil {
		panic(err)
	}
	return ds
}

func accuracy(model *tensai.Sequential, ds *tensai.Dataset) float64 {
	pred, err := model.Predict(ds.Inputs)
	if err != nil {
		panic(err)
	}
	correct := 0
	for r := 0; r < pred.Rows; r++ {
		best := 0
		for c := 1; c < pred.Cols; c++ {
			if pred.At(r, c) > pred.At(r, best) {
				best = c
			}
		}
		if best == int(ds.Targets.At(r, 0)) {
			correct++
		}
	}
	return float64(correct) / float64(ds.Len())
}

func main() {
	rng := rand.New(rand.NewSource(seed))
	ds := synthesize(rng)

	// The synthetic data alternates classes, so shuffle before splitting.
	ds.Shuffle(rng)
	train, test, err := ds.Split(0.2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("split: %d train / %d test samples\n", train.Len(), test.Len())

	// Fit normalization on the training split only, then apply the same
	// transform to the test split — never let test data leak into the stats.
	mean, std := train.Standardize()
	test.StandardizeWith(mean, std)
	fmt.Printf("train feature means: %.1f %.1f %.1f (std %.1f %.1f %.1f)\n",
		mean[0], mean[1], mean[2], std[0], std[1], std[2])

	model := tensai.NewSequential()
	model.Add(tensai.NewDense(16))
	model.Add(&tensai.ReLU{})
	model.Add(tensai.NewDense(classes))
	if err := model.Compile(features, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.01)); err != nil {
		panic(err)
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		var lossSum float32
		var steps int
		err := train.Batches(batchSize, rng, func(in, tgt *tensai.Matrix) error {
			loss, err := model.FitStep(in, tgt)
			lossSum += loss
			steps++
			return err
		})
		if err != nil {
			panic(err)
		}
		if epoch == 1 || epoch%20 == 0 {
			fmt.Printf("epoch %3d: loss=%.4f\n", epoch, lossSum/float32(steps))
		}
	}

	fmt.Printf("\ntrain accuracy: %.1f%%\n", accuracy(model, train)*100)
	fmt.Printf("test accuracy:  %.1f%%\n", accuracy(model, test)*100)
}
