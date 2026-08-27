// Command mnist trains a digit classifier on the MNIST dataset using the
// built-in dataset/mnist loader, which downloads the data into
// os.UserCacheDir()/tensai/mnist on first use.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset/mnist"
	tensaitflite "github.com/mattn/tensai/encoding/tflite"
	"github.com/mattn/tensai/knn"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/metrics"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const (
	classCount = mnist.ClassCount
	imageSize  = mnist.ImageSize
	trainLimit = 5000
	testLimit  = 1000
	epochs     = 4
	batchSize  = 100
	seed       = 5
)

// predictChunked runs Predict over the split in chunks: the CNN's im2col
// expansion over a whole split at once would be needlessly large.
func predictChunked(model model.Model, inputs *tensai.Matrix) (*tensai.Matrix, error) {
	out := tensai.NewMatrix(inputs.Rows, classCount)
	const chunk = 500
	for off := 0; off < inputs.Rows; off += chunk {
		n := min(chunk, inputs.Rows-off)
		view := &tensai.Matrix{Rows: n, Cols: imageSize, Data: inputs.Data[off*imageSize : (off+n)*imageSize]}
		pred, err := model.Predict(view)
		if err != nil {
			return nil, err
		}
		copy(out.Data[off*classCount:(off+n)*classCount], pred.Data)
	}
	return out, nil
}

func evaluate(model model.Model, inputs, targets *tensai.Matrix) (*metrics.Result, error) {
	pred, err := predictChunked(model, inputs)
	if err != nil {
		return nil, err
	}
	return metrics.Report(pred, targets)
}

// buildModel constructs either the plain MLP or a small CNN and compiles it.
func buildModel(kind string) (*model.Sequential, error) {
	model := model.NewSequential()
	switch kind {
	case "dense":
		model.Add(layer.NewDense(128))
		model.Add(&layer.ReLU{})
		model.Add(layer.NewDense(64))
		model.Add(&layer.ReLU{})
		model.Add(layer.NewDense(classCount))
	case "cnn":
		model.Add(layer.NewConv2D(8, 3, 1, 1)) // 28x28x1 -> 28x28x8
		model.Add(&layer.ReLU{})
		model.Add(layer.NewMaxPool2D(2)) // -> 14x14x8
		model.Add(layer.NewConv2D(16, 3, 1, 1))
		model.Add(&layer.ReLU{})
		model.Add(layer.NewMaxPool2D(2)) // -> 7x7x16
		model.Add(layer.NewDense(64))
		model.Add(&layer.ReLU{})
		model.Add(layer.NewDropout(0.25))
		model.Add(layer.NewDense(classCount))
	default:
		return nil, fmt.Errorf("unknown model kind %q (want dense or cnn)", kind)
	}
	var err error
	if kind == "cnn" {
		// The input geometry is stated once; Conv2D and MaxPool2D pick their
		// dimensions up from the threaded shape.
		err = model.CompileImage(layer.Image{H: 28, W: 28, C: 1}, loss.SoftmaxCrossEntropy{}, optim.NewAdamW(0.001, 0.01))
	} else {
		err = model.Compile(imageSize, loss.SoftmaxCrossEntropy{}, optim.NewAdamW(0.001, 0.01))
	}
	if err != nil {
		return nil, err
	}
	return model, nil
}

func main() {
	kind := flag.String("model", "dense", "model architecture: dense, cnn, or knn")
	export := flag.String("export", "", "write the trained model as TFLite to this path")
	flag.Parse()

	// MNIST_DIR overrides the default cache directory.
	data, err := mnist.Load(&mnist.Options{
		Dir:        os.Getenv("MNIST_DIR"),
		TrainLimit: trainLimit,
		TestLimit:  testLimit,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "mnist: %v\n", err)
		os.Exit(1)
	}
	train, test := data.Train, data.Test

	if *kind == "knn" {
		if *export != "" {
			fmt.Fprintln(os.Stderr, "mnist: -export is not supported with -model knn")
			os.Exit(1)
		}
		// A lazy baseline: no training at all, just neighbor votes. The
		// distance matrix runs on the same Dot kernel as the networks.
		knn := knn.New(3)
		if err := knn.Fit(train.Inputs, train.Targets); err != nil {
			panic(err)
		}
		res, err := evaluate(knn, test.Inputs, test.Targets)
		if err != nil {
			panic(err)
		}
		fmt.Printf("knn (k=3) test accuracy: %d/%d\n", res.Correct, res.Total)
		printConfusion(res.Confusion)
		return
	}

	model, err := buildModel(*kind)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mnist: %v\n", err)
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(seed))
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
		fmt.Printf("epoch %2d: loss=%.6f\n", epoch, lossSum/float32(steps))
	}

	trainRes, err := evaluate(model, train.Inputs, train.Targets)
	if err != nil {
		panic(err)
	}
	testRes, err := evaluate(model, test.Inputs, test.Targets)
	if err != nil {
		panic(err)
	}

	fmt.Printf("\ntrain accuracy: %d/%d\n", trainRes.Correct, trainRes.Total)
	fmt.Printf("test accuracy:  %d/%d\n", testRes.Correct, testRes.Total)

	// Round-trip the trained parameters through JSON to demonstrate
	// serialization: a freshly built model must score identically.
	modelDir := os.Getenv("MNIST_DIR")
	if modelDir == "" {
		if modelDir, err = mnist.DefaultDir(); err != nil {
			panic(err)
		}
	}
	modelPath := filepath.Join(modelDir, "model-"+*kind+".json")
	if err := model.SaveFile(modelPath); err != nil {
		panic(err)
	}
	reloaded, err := buildModel(*kind)
	if err != nil {
		panic(err)
	}
	if err := reloaded.LoadFile(modelPath); err != nil {
		panic(err)
	}
	reloadedRes, err := evaluate(reloaded, test.Inputs, test.Targets)
	if err != nil {
		panic(err)
	}
	fmt.Printf("saved to %s, reloaded accuracy: %d/%d\n", modelPath, reloadedRes.Correct, reloadedRes.Total)

	if *export != "" {
		// MNIST is single-channel, so the flat rows already match the NHWC
		// layout the exported model expects.
		if err := tensaitflite.MarshalFile(*export, model); err != nil {
			fmt.Fprintf(os.Stderr, "mnist: tflite export: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("exported TFLite model to %s\n", *export)
	}
	printConfusion(testRes.Confusion)
}

func printConfusion(confusion [][]int) {
	fmt.Println("\nconfusion matrix (rows = actual, columns = predicted):")
	for r := range confusion {
		fmt.Printf("  %d:", r)
		for _, n := range confusion[r] {
			fmt.Printf(" %4d", n)
		}
		fmt.Println()
	}
}
