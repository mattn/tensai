package main

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	tensai "github.com/mattn/tensai"
)

const (
	imageMagic     = 2051
	labelMagic     = 2049
	imageSize      = 28 * 28
	classCount     = 10
	trainLimit     = 5000
	testLimit      = 1000
	epochs         = 4
	batchSize      = 100
	defaultDataDir = "_example/mnist/data"
	seed           = 5
	mnistBaseURL   = "https://storage.googleapis.com/cvdf-datasets/mnist"
)

var mnistFiles = []string{
	"train-images-idx3-ubyte",
	"train-labels-idx1-ubyte",
	"t10k-images-idx3-ubyte",
	"t10k-labels-idx1-ubyte",
}

func exists(dir, name string) bool {
	for _, candidate := range []string{
		filepath.Join(dir, name),
		filepath.Join(dir, name+".gz"),
	} {
		if _, err := os.Stat(candidate); err == nil {
			return true
		}
	}
	return false
}

func ensureMNIST(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	for _, name := range mnistFiles {
		if exists(dir, name) {
			continue
		}
		dst := filepath.Join(dir, name+".gz")
		fmt.Printf("downloading %s\n", dst)
		if err := downloadFile(mnistBaseURL+"/"+name+".gz", dst); err != nil {
			return err
		}
	}
	return nil
}

func downloadFile(url, path string) error {
	client := http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}

	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, path)
}

func openMaybeGzip(dir, name string) (io.ReadCloser, string, error) {
	for _, candidate := range []string{
		filepath.Join(dir, name),
		filepath.Join(dir, name+".gz"),
	} {
		f, err := os.Open(candidate)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, "", err
		}
		if filepath.Ext(candidate) != ".gz" {
			return f, candidate, nil
		}
		gz, err := gzip.NewReader(f)
		if err != nil {
			_ = f.Close()
			return nil, "", err
		}
		return struct {
			io.Reader
			io.Closer
		}{Reader: gz, Closer: multiCloser{gz, f}}, candidate, nil
	}
	return nil, "", fmt.Errorf("missing %s or %s.gz in %s", name, name, dir)
}

type multiCloser []io.Closer

func (m multiCloser) Close() error {
	var first error
	for _, c := range m {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func readImages(dir, name string, limit int) (*tensai.Matrix, error) {
	r, path, err := openMaybeGzip(dir, name)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var magic, count, rows, cols uint32
	for _, field := range []*uint32{&magic, &count, &rows, &cols} {
		if err := binary.Read(r, binary.BigEndian, field); err != nil {
			return nil, err
		}
	}
	if magic != imageMagic || rows*cols != imageSize {
		return nil, fmt.Errorf("%s: bad image header magic=%d shape=%dx%d", path, magic, rows, cols)
	}
	if limit <= 0 || limit > int(count) {
		limit = int(count)
	}

	inputs := tensai.NewMatrix(limit, imageSize)
	buf := make([]byte, imageSize)
	for row := 0; row < limit; row++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		for i, px := range buf {
			inputs.Data[row*imageSize+i] = float32(px) / 255
		}
	}
	return inputs, nil
}

func readLabels(dir, name string, limit int) (*tensai.Matrix, error) {
	r, path, err := openMaybeGzip(dir, name)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var magic, count uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.BigEndian, &count); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, fmt.Errorf("%s: bad label header magic=%d", path, magic)
	}
	if limit <= 0 || limit > int(count) {
		limit = int(count)
	}

	targets := tensai.NewMatrix(limit, 1)
	buf := make([]byte, limit)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	for i, label := range buf {
		if label >= classCount {
			return nil, fmt.Errorf("%s: label %d out of range", path, label)
		}
		targets.Data[i] = float32(label)
	}
	return targets, nil
}

func loadMNIST(dir string) (*tensai.Matrix, *tensai.Matrix, *tensai.Matrix, *tensai.Matrix, error) {
	trainImages, err := readImages(dir, "train-images-idx3-ubyte", trainLimit)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	trainLabels, err := readLabels(dir, "train-labels-idx1-ubyte", trainImages.Rows)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	testImages, err := readImages(dir, "t10k-images-idx3-ubyte", testLimit)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	testLabels, err := readLabels(dir, "t10k-labels-idx1-ubyte", testImages.Rows)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return trainImages, trainLabels, testImages, testLabels, nil
}

func fillBatch(inputs, targets, batchInputs, batchTargets *tensai.Matrix, rows []int) {
	for i, r := range rows {
		copy(batchInputs.Data[i*imageSize:(i+1)*imageSize], inputs.Data[r*imageSize:(r+1)*imageSize])
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

func evaluate(model tensai.Model, inputs, targets *tensai.Matrix) (int, [classCount][classCount]int, error) {
	var confusion [classCount][classCount]int
	correct := 0
	// Predict in chunks: the CNN's im2col expansion over a whole split at
	// once would be needlessly large.
	const chunk = 500
	for off := 0; off < inputs.Rows; off += chunk {
		n := min(chunk, inputs.Rows-off)
		view := &tensai.Matrix{Rows: n, Cols: imageSize, Data: inputs.Data[off*imageSize : (off+n)*imageSize]}
		pred, err := model.Predict(view)
		if err != nil {
			return 0, confusion, err
		}
		for r := 0; r < n; r++ {
			got := argmaxRow(pred, r)
			want := int(targets.At(off+r, 0))
			confusion[want][got]++
			if got == want {
				correct++
			}
		}
	}
	return correct, confusion, nil
}

// buildModel constructs either the plain MLP or a small CNN and compiles it.
func buildModel(kind string) (*tensai.Sequential, error) {
	model := tensai.NewSequential()
	switch kind {
	case "dense":
		model.Add(tensai.NewDense(128))
		model.Add(&tensai.ReLU{})
		model.Add(tensai.NewDense(64))
		model.Add(&tensai.ReLU{})
		model.Add(tensai.NewDense(classCount))
	case "cnn":
		model.Add(tensai.NewConv2D(28, 28, 1, 8, 3, 1, 1)) // 28x28x1 -> 28x28x8
		model.Add(&tensai.ReLU{})
		model.Add(tensai.NewMaxPool2D(28, 28, 8, 2)) // -> 14x14x8
		model.Add(tensai.NewConv2D(14, 14, 8, 16, 3, 1, 1))
		model.Add(&tensai.ReLU{})
		model.Add(tensai.NewMaxPool2D(14, 14, 16, 2)) // -> 7x7x16
		model.Add(tensai.NewDense(64))
		model.Add(&tensai.ReLU{})
		model.Add(tensai.NewDropout(0.25))
		model.Add(tensai.NewDense(classCount))
	default:
		return nil, fmt.Errorf("unknown model kind %q (want dense or cnn)", kind)
	}
	if err := model.Compile(imageSize, tensai.SoftmaxCrossEntropy{}, tensai.NewAdamW(0.001, 0.01)); err != nil {
		return nil, err
	}
	return model, nil
}

func main() {
	kind := flag.String("model", "dense", "model architecture: dense, cnn, or knn")
	flag.Parse()

	dataDir := os.Getenv("MNIST_DIR")
	if dataDir == "" {
		dataDir = defaultDataDir
	}
	if err := ensureMNIST(dataDir); err != nil {
		fmt.Fprintf(os.Stderr, "mnist: download failed: %v\n", err)
		os.Exit(1)
	}
	trainInputs, trainTargets, testInputs, testTargets, err := loadMNIST(dataDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mnist: %v\n", err)
		os.Exit(1)
	}

	if *kind == "knn" {
		// A lazy baseline: no training at all, just neighbor votes. The
		// distance matrix runs on the same Dot kernel as the networks.
		knn := tensai.NewKNN(3)
		if err := knn.Fit(trainInputs, trainTargets); err != nil {
			panic(err)
		}
		correct, confusion, err := evaluate(knn, testInputs, testTargets)
		if err != nil {
			panic(err)
		}
		fmt.Printf("knn (k=3) test accuracy: %d/%d\n", correct, testInputs.Rows)
		printConfusion(confusion)
		return
	}

	model, err := buildModel(*kind)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mnist: %v\n", err)
		os.Exit(1)
	}

	batchInputs := tensai.NewMatrix(batchSize, imageSize)
	batchTargets := tensai.NewMatrix(batchSize, 1)
	rng := rand.New(rand.NewSource(seed))
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
		fmt.Printf("epoch %2d: loss=%.6f\n", epoch, lossSum/float32(steps))
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

	// Round-trip the trained parameters through JSON to demonstrate
	// serialization: a freshly built model must score identically.
	modelPath := filepath.Join(dataDir, "model-"+*kind+".json")
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
	reloadedCorrect, _, err := evaluate(reloaded, testInputs, testTargets)
	if err != nil {
		panic(err)
	}
	fmt.Printf("saved to %s, reloaded accuracy: %d/%d\n", modelPath, reloadedCorrect, testInputs.Rows)
	printConfusion(confusion)
}

func printConfusion(confusion [classCount][classCount]int) {
	fmt.Println("\nconfusion matrix (rows = actual, columns = predicted):")
	for r := 0; r < classCount; r++ {
		fmt.Printf("  %d:", r)
		for c := 0; c < classCount; c++ {
			fmt.Printf(" %4d", confusion[r][c])
		}
		fmt.Println()
	}
}
