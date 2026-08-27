// Package iris downloads, caches, and loads Fisher's iris dataset as a
// ready-to-use Dataset.
//
// The CSV (about 4KB) is fetched from the UCI Machine Learning Repository
// on first use and cached under os.UserCacheDir()/tensai/iris
// (~/.cache/tensai/iris on Linux); subsequent loads read from the cache
// without touching the network.
package iris

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/dataset"
	"github.com/mattn/tensai/dataset/internal/fetch"
)

const (
	// FeatureCount is the number of input features per sample.
	FeatureCount = 4
	// ClassCount is the number of species.
	ClassCount = 3
	// SampleCount is the total number of samples.
	SampleCount = 150

	url      = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	fileName = "iris.data"
)

// FeatureNames names the input columns, in order.
var FeatureNames = []string{"sepal length", "sepal width", "petal length", "petal width"}

// ClassNames maps a class index to its species name.
var ClassNames = []string{"setosa", "versicolor", "virginica"}

// Options configures Load. The zero value (or a nil pointer) loads from
// the default cache directory.
type Options struct {
	// Dir is the directory holding (or receiving) the CSV file. Empty
	// means DefaultDir().
	Dir string
}

// DefaultDir returns the default cache directory,
// os.UserCacheDir()/tensai/iris.
func DefaultDir() (string, error) {
	return fetch.DefaultDir("iris")
}

// Load returns the iris dataset, downloading it into the cache directory
// first if it is not already present: inputs of shape 150x4 (centimeters)
// and targets of shape 150x1 holding the class index. Rows are grouped by
// class; shuffle or split before training as needed.
func Load(opt *Options) (*dataset.Dataset, error) {
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
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, fileName)
	if _, err := os.Stat(path); err != nil {
		if err := fetch.Download(url, path); err != nil {
			return nil, err
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parse(path, string(raw))
}

func parse(path, raw string) (*dataset.Dataset, error) {
	classes := map[string]int{}
	for i, name := range ClassNames {
		classes["Iris-"+name] = i
	}
	inputs := tensai.NewMatrix(SampleCount, FeatureCount)
	targets := tensai.NewMatrix(SampleCount, 1)
	row := 0
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if row >= SampleCount {
			return nil, fmt.Errorf("tensai: iris: %s: more than %d rows", path, SampleCount)
		}
		fields := strings.Split(line, ",")
		if len(fields) != FeatureCount+1 {
			return nil, fmt.Errorf("tensai: iris: %s: bad row %q", path, line)
		}
		for c := 0; c < FeatureCount; c++ {
			v, err := strconv.ParseFloat(fields[c], 32)
			if err != nil {
				return nil, fmt.Errorf("tensai: iris: %s: bad row %q: %w", path, line, err)
			}
			inputs.Set(row, c, tensai.Float(v))
		}
		cls, ok := classes[fields[FeatureCount]]
		if !ok {
			return nil, fmt.Errorf("tensai: iris: %s: unknown class %q", path, fields[FeatureCount])
		}
		targets.Set(row, 0, tensai.Float(cls))
		row++
	}
	if row != SampleCount {
		return nil, fmt.Errorf("tensai: iris: %s: %d rows, want %d", path, row, SampleCount)
	}
	return dataset.New(inputs, targets)
}
