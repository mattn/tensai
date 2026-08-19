package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

	tensai "github.com/mattn/tensai"
)

const (
	featureCount  = 4
	classCount    = 3
	trainPerClass = 40
	epochs        = 900
	batchSize     = 32
	seed          = 3
)

type sample struct {
	x     [featureCount]float32
	class int
}

func parseData() ([]sample, error) {
	lines := strings.Split(strings.TrimSpace(irisCSV), "\n")
	data := make([]sample, 0, len(lines))
	for _, line := range lines {
		fields := strings.Split(line, ",")
		if len(fields) != featureCount+1 {
			return nil, fmt.Errorf("bad row: %q", line)
		}
		var s sample
		for i := 0; i < featureCount; i++ {
			v, err := strconv.ParseFloat(fields[i], 64)
			if err != nil {
				return nil, err
			}
			s.x[i] = float32(v)
		}
		cls, err := strconv.Atoi(fields[featureCount])
		if err != nil {
			return nil, err
		}
		s.class = cls
		data = append(data, s)
	}
	return data, nil
}

func split(data []sample) (train, test []sample) {
	rng := rand.New(rand.NewSource(seed))
	for cls := 0; cls < classCount; cls++ {
		var rows []sample
		for _, s := range data {
			if s.class == cls {
				rows = append(rows, s)
			}
		}
		rng.Shuffle(len(rows), func(i, j int) {
			rows[i], rows[j] = rows[j], rows[i]
		})
		train = append(train, rows[:trainPerClass]...)
		test = append(test, rows[trainPerClass:]...)
	}
	return train, test
}

func standardize(train, test []sample) {
	var mean, variance [featureCount]float32
	for _, s := range train {
		for c, v := range s.x {
			mean[c] += v
		}
	}
	for c := range mean {
		mean[c] /= float32(len(train))
	}
	for _, s := range train {
		for c, v := range s.x {
			diff := v - mean[c]
			variance[c] += diff * diff
		}
	}
	for c := range variance {
		variance[c] /= float32(len(train))
		if variance[c] == 0 {
			variance[c] = 1
		}
	}
	apply := func(rows []sample) {
		for i := range rows {
			for c := range rows[i].x {
				rows[i].x[c] = (rows[i].x[c] - mean[c]) / float32(math.Sqrt(float64(variance[c])))
			}
		}
	}
	apply(train)
	apply(test)
}

func matrices(data []sample) (*tensai.Matrix, *tensai.Matrix) {
	inputs := tensai.NewMatrix(len(data), featureCount)
	targets := tensai.NewMatrix(len(data), 1)
	for r, s := range data {
		for c, v := range s.x {
			inputs.Set(r, c, v)
		}
		targets.Set(r, 0, float32(s.class))
	}
	return inputs, targets
}

func fillBatch(inputs, targets, batchInputs, batchTargets *tensai.Matrix, rows []int) {
	for i, r := range rows {
		copy(batchInputs.Data[i*featureCount:(i+1)*featureCount], inputs.Data[r*featureCount:(r+1)*featureCount])
		batchTargets.Data[i] = targets.Data[r]
	}
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

func evaluate(model *tensai.Sequential, inputs, targets *tensai.Matrix) (int, [classCount][classCount]int, error) {
	pred, err := model.Predict(inputs)
	if err != nil {
		return 0, [classCount][classCount]int{}, err
	}
	var confusion [classCount][classCount]int
	correct := 0
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
	data, err := parseData()
	if err != nil {
		panic(err)
	}
	train, test := split(data)
	standardize(train, test)
	trainInputs, trainTargets := matrices(train)
	testInputs, testTargets := matrices(test)

	model := tensai.NewSequential()
	model.Add(tensai.NewDense(12))
	model.Add(&tensai.Tanh{})
	model.Add(tensai.NewDense(classCount))
	if err := model.Compile(featureCount, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.03)); err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(seed + 1))
	batchInputs := tensai.NewMatrix(batchSize, featureCount)
	batchTargets := tensai.NewMatrix(batchSize, 1)
	for epoch := 1; epoch <= epochs; epoch++ {
		perm := rng.Perm(trainInputs.Rows)
		var lossSum float32
		var steps int
		for off := 0; off+batchSize <= len(perm); off += batchSize {
			fillBatch(trainInputs, trainTargets, batchInputs, batchTargets, perm[off:off+batchSize])
			loss, err := model.FitStep(batchInputs, batchTargets)
			if err != nil {
				panic(err)
			}
			lossSum += loss
			steps++
		}
		if epoch == 1 || epoch%150 == 0 || epoch == epochs {
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
	for r := 0; r < classCount; r++ {
		fmt.Printf("  class %d: %3d %3d %3d\n", r, confusion[r][0], confusion[r][1], confusion[r][2])
	}
}

const irisCSV = `
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.4,3.7,1.5,0.2,0
4.8,3.4,1.6,0.2,0
4.8,3.0,1.4,0.1,0
4.3,3.0,1.1,0.1,0
5.8,4.0,1.2,0.2,0
5.7,4.4,1.5,0.4,0
5.4,3.9,1.3,0.4,0
5.1,3.5,1.4,0.3,0
5.7,3.8,1.7,0.3,0
5.1,3.8,1.5,0.3,0
5.4,3.4,1.7,0.2,0
5.1,3.7,1.5,0.4,0
4.6,3.6,1.0,0.2,0
5.1,3.3,1.7,0.5,0
4.8,3.4,1.9,0.2,0
5.0,3.0,1.6,0.2,0
5.0,3.4,1.6,0.4,0
5.2,3.5,1.5,0.2,0
5.2,3.4,1.4,0.2,0
4.7,3.2,1.6,0.2,0
4.8,3.1,1.6,0.2,0
5.4,3.4,1.5,0.4,0
5.2,4.1,1.5,0.1,0
5.5,4.2,1.4,0.2,0
4.9,3.1,1.5,0.2,0
5.0,3.2,1.2,0.2,0
5.5,3.5,1.3,0.2,0
4.9,3.6,1.4,0.1,0
4.4,3.0,1.3,0.2,0
5.1,3.4,1.5,0.2,0
5.0,3.5,1.3,0.3,0
4.5,2.3,1.3,0.3,0
4.4,3.2,1.3,0.2,0
5.0,3.5,1.6,0.6,0
5.1,3.8,1.9,0.4,0
4.8,3.0,1.4,0.3,0
5.1,3.8,1.6,0.2,0
4.6,3.2,1.4,0.2,0
5.3,3.7,1.5,0.2,0
5.0,3.3,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,2.8,4.6,1.5,1
5.7,2.8,4.5,1.3,1
6.3,3.3,4.7,1.6,1
4.9,2.4,3.3,1.0,1
6.6,2.9,4.6,1.3,1
5.2,2.7,3.9,1.4,1
5.0,2.0,3.5,1.0,1
5.9,3.0,4.2,1.5,1
6.0,2.2,4.0,1.0,1
6.1,2.9,4.7,1.4,1
5.6,2.9,3.6,1.3,1
6.7,3.1,4.4,1.4,1
5.6,3.0,4.5,1.5,1
5.8,2.7,4.1,1.0,1
6.2,2.2,4.5,1.5,1
5.6,2.5,3.9,1.1,1
5.9,3.2,4.8,1.8,1
6.1,2.8,4.0,1.3,1
6.3,2.5,4.9,1.5,1
6.1,2.8,4.7,1.2,1
6.4,2.9,4.3,1.3,1
6.6,3.0,4.4,1.4,1
6.8,2.8,4.8,1.4,1
6.7,3.0,5.0,1.7,1
6.0,2.9,4.5,1.5,1
5.7,2.6,3.5,1.0,1
5.5,2.4,3.8,1.1,1
5.5,2.4,3.7,1.0,1
5.8,2.7,3.9,1.2,1
6.0,2.7,5.1,1.6,1
5.4,3.0,4.5,1.5,1
6.0,3.4,4.5,1.6,1
6.7,3.1,4.7,1.5,1
6.3,2.3,4.4,1.3,1
5.6,3.0,4.1,1.3,1
5.5,2.5,4.0,1.3,1
5.5,2.6,4.4,1.2,1
6.1,3.0,4.6,1.4,1
5.8,2.6,4.0,1.2,1
5.0,2.3,3.3,1.0,1
5.6,2.7,4.2,1.3,1
5.7,3.0,4.2,1.2,1
5.7,2.9,4.2,1.3,1
6.2,2.9,4.3,1.3,1
5.1,2.5,3.0,1.1,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3.0,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3.0,5.8,2.2,2
7.6,3.0,6.6,2.1,2
4.9,2.5,4.5,1.7,2
7.3,2.9,6.3,1.8,2
6.7,2.5,5.8,1.8,2
7.2,3.6,6.1,2.5,2
6.5,3.2,5.1,2.0,2
6.4,2.7,5.3,1.9,2
6.8,3.0,5.5,2.1,2
5.7,2.5,5.0,2.0,2
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3.0,5.5,1.8,2
7.7,3.8,6.7,2.2,2
7.7,2.6,6.9,2.3,2
6.0,2.2,5.0,1.5,2
6.9,3.2,5.7,2.3,2
5.6,2.8,4.9,2.0,2
7.7,2.8,6.7,2.0,2
6.3,2.7,4.9,1.8,2
6.7,3.3,5.7,2.1,2
7.2,3.2,6.0,1.8,2
6.2,2.8,4.8,1.8,2
6.1,3.0,4.9,1.8,2
6.4,2.8,5.6,2.1,2
7.2,3.0,5.8,1.6,2
7.4,2.8,6.1,1.9,2
7.9,3.8,6.4,2.0,2
6.4,2.8,5.6,2.2,2
6.3,2.8,5.1,1.5,2
6.1,2.6,5.6,1.4,2
7.7,3.0,6.1,2.3,2
6.3,3.4,5.6,2.4,2
6.4,3.1,5.5,1.8,2
6.0,3.0,4.8,1.8,2
6.9,3.1,5.4,2.1,2
6.7,3.1,5.6,2.4,2
6.9,3.1,5.1,2.3,2
5.8,2.7,5.1,1.9,2
6.8,3.2,5.9,2.3,2
6.7,3.3,5.7,2.5,2
6.7,3.0,5.2,2.3,2
6.3,2.5,5.0,1.9,2
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2
`
