// Package mnist downloads, caches, and loads the MNIST handwritten digit
// dataset as ready-to-use Datasets.
//
// The four IDX files (about 11MB compressed) are fetched on first use and
// cached under os.UserCacheDir()/tensai/mnist (~/.cache/tensai/mnist on
// Linux); subsequent loads read from the cache without touching the
// network.
package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset"
	"github.com/mattn/tensai/dataset/internal/fetch"
)

const (
	// ImageWidth and ImageHeight are the pixel dimensions of each image.
	ImageWidth  = 28
	ImageHeight = 28
	// ImageSize is the number of pixels (and input features) per image.
	ImageSize = ImageWidth * ImageHeight
	// ClassCount is the number of digit classes.
	ClassCount = 10

	baseURL    = "https://storage.googleapis.com/cvdf-datasets/mnist"
	imageMagic = 2051
	labelMagic = 2049
)

var files = []string{
	"train-images-idx3-ubyte",
	"train-labels-idx1-ubyte",
	"t10k-images-idx3-ubyte",
	"t10k-labels-idx1-ubyte",
}

// Options configures Load. The zero value (or a nil pointer) loads the
// full dataset from the default cache directory.
type Options struct {
	// Dir is the directory holding (or receiving) the IDX files. Empty
	// means DefaultDir().
	Dir string
	// TrainLimit caps the number of training samples; 0 loads all 60000.
	TrainLimit int
	// TestLimit caps the number of test samples; 0 loads all 10000.
	TestLimit int
}

// Data holds the two MNIST splits. Inputs are ImageSize columns of pixel
// intensities scaled to [0, 1]; targets are one column with the digit.
type Data struct {
	Train *dataset.Dataset
	Test  *dataset.Dataset
}

// DefaultDir returns the default cache directory,
// os.UserCacheDir()/tensai/mnist.
func DefaultDir() (string, error) {
	return fetch.DefaultDir("mnist")
}

// Load returns the MNIST dataset, downloading it into the cache directory
// first if it is not already present.
func Load(opt *Options) (*Data, error) {
	if opt == nil {
		opt = &Options{}
	}
	dir := opt.Dir
	if dir == "" {
		var err error
		if dir, err = DefaultDir(); err != nil {
			return nil, err
		}
	}
	if err := ensure(dir); err != nil {
		return nil, err
	}
	trainInputs, err := readImages(dir, files[0], opt.TrainLimit)
	if err != nil {
		return nil, err
	}
	trainTargets, err := readLabels(dir, files[1], trainInputs.Rows)
	if err != nil {
		return nil, err
	}
	testInputs, err := readImages(dir, files[2], opt.TestLimit)
	if err != nil {
		return nil, err
	}
	testTargets, err := readLabels(dir, files[3], testInputs.Rows)
	if err != nil {
		return nil, err
	}
	train, err := dataset.New(trainInputs, trainTargets)
	if err != nil {
		return nil, err
	}
	test, err := dataset.New(testInputs, testTargets)
	if err != nil {
		return nil, err
	}
	return &Data{Train: train, Test: test}, nil
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

func ensure(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	for _, name := range files {
		if exists(dir, name) {
			continue
		}
		if err := fetch.Download(baseURL+"/"+name+".gz", filepath.Join(dir, name+".gz")); err != nil {
			return err
		}
	}
	return nil
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
	return nil, "", fmt.Errorf("tensai: mnist: missing %s or %s.gz in %s", name, name, dir)
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
	if magic != imageMagic || rows*cols != ImageSize {
		return nil, fmt.Errorf("tensai: mnist: %s: bad image header magic=%d shape=%dx%d", path, magic, rows, cols)
	}
	if limit <= 0 || limit > int(count) {
		limit = int(count)
	}

	inputs := tensai.NewMatrix(limit, ImageSize)
	buf := make([]byte, ImageSize)
	for row := 0; row < limit; row++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}
		for i, px := range buf {
			inputs.Data[row*ImageSize+i] = tensai.Float(px) / 255
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
		return nil, fmt.Errorf("tensai: mnist: %s: bad label header magic=%d", path, magic)
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
		if label >= ClassCount {
			return nil, fmt.Errorf("tensai: mnist: %s: label %d out of range", path, label)
		}
		targets.Data[i] = tensai.Float(label)
	}
	return targets, nil
}
