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
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const (
	trainPerClass = 40
	epochs        = 900
	batchSize     = 32
	seed          = 3
)

// splitStratified shuffles each class independently and keeps
// trainPerClass rows of every class in the training split, so both splits
// preserve the 1/3-per-class balance.
func splitStratified(ds *dataset.Dataset, rng *rand.Rand) (train, test *dataset.Dataset, err error) {
	var trainIdx, testIdx []int
	for cls := 0; cls < iris.ClassCount; cls++ {
		var rows []int
		for r := 0; r < ds.Len(); r++ {
			if int(ds.Targets.At(r, 0)) == cls {
				rows = append(rows, r)
			}
		}
		rng.Shuffle(len(rows), func(i, j int) {
			rows[i], rows[j] = rows[j], rows[i]
		})
		trainIdx = append(trainIdx, rows[:trainPerClass]...)
		testIdx = append(testIdx, rows[trainPerClass:]...)
	}
	pick := func(idx []int) (*dataset.Dataset, error) {
		inputs := tensai.NewMatrix(len(idx), ds.Inputs.Cols)
		targets := tensai.NewMatrix(len(idx), 1)
		for i, r := range idx {
			for c := 0; c < ds.Inputs.Cols; c++ {
				inputs.Set(i, c, ds.Inputs.At(r, c))
			}
			targets.Set(i, 0, ds.Targets.At(r, 0))
		}
		return dataset.New(inputs, targets)
	}
	if train, err = pick(trainIdx); err != nil {
		return nil, nil, err
	}
	if test, err = pick(testIdx); err != nil {
		return nil, nil, err
	}
	return train, test, nil
}

func argmaxRow(m *tensai.Matrix, row int) int {
	best := 0
	for c := 1; c < m.Cols; c++ {
		if m.At(row, c) > m.At(row, best) {
			best = c
		}
	}
	return best
}

func evaluate(net *model.Sequential, ds *dataset.Dataset) (int, [iris.ClassCount][iris.ClassCount]int, error) {
	pred, err := net.Predict(ds.Inputs)
	if err != nil {
		return 0, [iris.ClassCount][iris.ClassCount]int{}, err
	}
	var confusion [iris.ClassCount][iris.ClassCount]int
	correct := 0
	for r := 0; r < pred.Rows; r++ {
		got := argmaxRow(pred, r)
		want := int(ds.Targets.At(r, 0))
		confusion[want][got]++
		if got == want {
			correct++
		}
	}
	return correct, confusion, nil
}

func main() {
	rng := rand.New(rand.NewSource(seed))
	train, test, err := splitStratified(iris.Load(), rng)
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
	for r := 0; r < iris.ClassCount; r++ {
		fmt.Printf("  %-10s:", iris.ClassNames[r])
		for c := 0; c < iris.ClassCount; c++ {
			fmt.Printf(" %3d", confusion[r][c])
		}
		fmt.Println()
	}
}
