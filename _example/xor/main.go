package main

import (
	"fmt"

	tensai "github.com/mattn/tensai"
)

func main() {
	// Classic XOR problem: a 2-input network with one hidden layer can learn
	// it when a non-linear activation is used.
	model := tensai.NewSequential()
	model.Add(tensai.NewDense(8))
	model.Add(&tensai.Tanh{})
	model.Add(tensai.NewDense(1))
	model.Add(&tensai.Sigmoid{})

	if err := model.Compile(2, tensai.MeanSquaredError{}, tensai.NewAdam(0.05)); err != nil {
		panic(err)
	}

	// XOR truth table: 4 samples, 2 features each.
	inputs, err := tensai.NewMatrixFromSlice(4, 2, []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	if err != nil {
		panic(err)
	}

	targets, err := tensai.NewMatrixFromSlice(4, 1, []float32{0, 1, 1, 0})
	if err != nil {
		panic(err)
	}

	if err := model.Fit(inputs, targets, 5000); err != nil {
		panic(err)
	}

	fmt.Println("\nXOR predictions after training:")
	pred, err := model.Predict(inputs)
	if err != nil {
		panic(err)
	}
	for r := 0; r < inputs.Rows; r++ {
		fmt.Printf("  [%g %g] -> %.4f (target %g)\n",
			inputs.At(r, 0), inputs.At(r, 1), pred.At(r, 0), targets.At(r, 0))
	}

	// Quick sanity check that the framework also works for classification
	// with softmax + cross-entropy on the same XOR inputs as 2 classes.
	fmt.Println("\nSoftmax + cross-entropy sanity check:")
	cls := tensai.NewSequential()
	cls.Add(tensai.NewDense(8))
	cls.Add(&tensai.ReLU{})
	cls.Add(tensai.NewDense(2))
	if err := cls.Compile(2, tensai.SoftmaxCrossEntropy{}, tensai.NewAdam(0.05)); err != nil {
		panic(err)
	}
	clsTargets, err := tensai.NewMatrixFromSlice(4, 1, []float32{
		0, // 0 XOR 0 -> class 0
		1, // 0 XOR 1 -> class 1
		1, // 1 XOR 0 -> class 1
		0, // 1 XOR 1 -> class 0
	})
	if err != nil {
		panic(err)
	}

	// Full-batch FitStep loop, demonstrating the lower-level training API
	// (the same thing model.Fit wraps for regression above).
	for epoch := 0; epoch < 5000; epoch++ {
		if _, err := cls.FitStep(inputs, clsTargets); err != nil {
			panic(err)
		}
	}
	clsPred, err := cls.Predict(inputs)
	if err != nil {
		panic(err)
	}
	for r := 0; r < inputs.Rows; r++ {
		// Softmax is applied inside the loss; raw logits come back from
		// Predict, so we report argmax as the predicted class.
		best := 0
		for c := 1; c < clsPred.Cols; c++ {
			if clsPred.At(r, c) > clsPred.At(r, best) {
				best = c
			}
		}
		fmt.Printf("  [%g %g] -> class %d (target %g)\n",
			inputs.At(r, 0), inputs.At(r, 1), best, clsTargets.At(r, 0))
	}
}
