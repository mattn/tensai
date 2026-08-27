package main

import (
	"fmt"
	"math"
	"math/rand"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const (
	numClasses     = 3
	featuresPerRow = 2
	pointsPerClass = 120
	trainPerClass  = 90
	hiddenWidth    = 64
	epochs         = 700
	batchSize      = 64
	turns          = 3.5
	noise          = 0.18
	trainingSeed   = 1
)

type sample struct {
	x0, x1 float32
	class  int
}

func makeSpiral() []sample {
	rng := rand.New(rand.NewSource(trainingSeed))
	data := make([]sample, 0, numClasses*pointsPerClass)
	for cls := 0; cls < numClasses; cls++ {
		for i := 0; i < pointsPerClass; i++ {
			r := float64(i) / float64(pointsPerClass-1)
			theta := float64(cls)*2*math.Pi/numClasses + turns*r + rng.NormFloat64()*noise
			data = append(data, sample{
				x0:    float32(r * math.Cos(theta)),
				x1:    float32(r * math.Sin(theta)),
				class: cls,
			})
		}
	}
	return data
}

func split(data []sample) (train, test []sample) {
	rng := rand.New(rand.NewSource(trainingSeed + 2))
	for cls := 0; cls < numClasses; cls++ {
		start := cls * pointsPerClass
		rows := append([]sample(nil), data[start:start+pointsPerClass]...)
		rng.Shuffle(len(rows), func(i, j int) {
			rows[i], rows[j] = rows[j], rows[i]
		})
		train = append(train, rows[:trainPerClass]...)
		test = append(test, rows[trainPerClass:]...)
	}
	return train, test
}

func matrices(data []sample) (*tensai.Matrix, *tensai.Matrix, error) {
	inputs := tensai.NewMatrix(len(data), featuresPerRow)
	targets := tensai.NewMatrix(len(data), 1)
	for r, s := range data {
		inputs.Set(r, 0, s.x0)
		inputs.Set(r, 1, s.x1)
		targets.Set(r, 0, float32(s.class))
	}
	return inputs, targets, nil
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

func evaluate(model *model.Sequential, inputs, targets *tensai.Matrix) (int, [numClasses][numClasses]int, error) {
	pred, err := model.Predict(inputs)
	if err != nil {
		return 0, [numClasses][numClasses]int{}, err
	}

	correct := 0
	var confusion [numClasses][numClasses]int
	for r := 0; r < pred.Rows; r++ {
		got := argmaxRow(pred, r)
		want := int(targets.At(r, 0))
		confusion[want][got]++
		if got == want {
			correct++
		}
	}
	return correct, confusion, nil
}

func main() {
	data := makeSpiral()
	train, test := split(data)
	trainInputs, trainTargets, err := matrices(train)
	if err != nil {
		panic(err)
	}
	testInputs, testTargets, err := matrices(test)
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
	ds, err := dataset.New(trainInputs, trainTargets)
	if err != nil {
		panic(err)
	}
	for epoch := 1; epoch <= epochs; epoch++ {
		var lossSum float32
		var steps int
		err := ds.Batches(batchSize, rng, func(in, tgt *tensai.Matrix) error {
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

	trainCorrect, _, err := evaluate(model, trainInputs, trainTargets)
	if err != nil {
		panic(err)
	}
	testCorrect, confusion, err := evaluate(model, testInputs, testTargets)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\ntrain accuracy: %d/%d\n", trainCorrect, trainInputs.Rows)
	fmt.Printf("test accuracy:  %d/%d\n", testCorrect, testInputs.Rows)
	fmt.Println("\nconfusion matrix (rows = actual, columns = predicted):")
	for r := 0; r < numClasses; r++ {
		fmt.Printf("  class %d: %3d %3d %3d\n", r, confusion[r][0], confusion[r][1], confusion[r][2])
	}
}
