package iris

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/mattn/tensai"
)

// writeFakeIris fills dir with a synthetic UCI-format iris.data: 50 rows
// per class whose sepal length encodes the row index, so tests never
// touch the network. The first row mirrors the real dataset.
func writeFakeIris(t *testing.T, dir string) {
	t.Helper()
	var b strings.Builder
	b.WriteString("5.1,3.5,1.4,0.2,Iris-setosa\n")
	for i := 1; i < SampleCount; i++ {
		fmt.Fprintf(&b, "%.1f,3.0,1.4,0.2,Iris-%s\n", float64(i)/10, ClassNames[i/(SampleCount/ClassCount)])
	}
	b.WriteString("\n") // the real file ends with a blank line
	if err := os.WriteFile(filepath.Join(dir, fileName), []byte(b.String()), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadFromDir(t *testing.T) {
	dir := t.TempDir()
	writeFakeIris(t, dir)

	ds, err := Load(&Options{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if ds.Inputs.Rows != SampleCount || ds.Inputs.Cols != FeatureCount {
		t.Fatalf("inputs shape = %dx%d, want %dx%d", ds.Inputs.Rows, ds.Inputs.Cols, SampleCount, FeatureCount)
	}
	if ds.Targets.Rows != SampleCount || ds.Targets.Cols != 1 {
		t.Fatalf("targets shape = %dx%d, want %dx1", ds.Targets.Rows, ds.Targets.Cols, SampleCount)
	}
	first := []tensai.Float{5.1, 3.5, 1.4, 0.2}
	for c, want := range first {
		if got := ds.Inputs.At(0, c); got != want {
			t.Fatalf("row 0 col %d = %g, want %g", c, got, want)
		}
	}
	var perClass [ClassCount]int
	for r := 0; r < ds.Len(); r++ {
		perClass[int(ds.Targets.At(r, 0))]++
	}
	for cls, n := range perClass {
		if n != SampleCount/ClassCount {
			t.Fatalf("class %d has %d samples, want %d", cls, n, SampleCount/ClassCount)
		}
	}
}

func TestLoadRejectsCorruptData(t *testing.T) {
	for name, content := range map[string]string{
		"bad float":     "x,3.5,1.4,0.2,Iris-setosa\n",
		"bad fields":    "5.1,3.5,1.4,Iris-setosa\n",
		"unknown class": "5.1,3.5,1.4,0.2,Iris-gigantea\n",
		"short file":    "5.1,3.5,1.4,0.2,Iris-setosa\n5.0,3.0,1.4,0.2,Iris-setosa\n",
	} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, fileName), []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(&Options{Dir: dir}); err == nil {
			t.Fatalf("%s did not error", name)
		}
	}
}

func TestDefaultDirUsesUserCache(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("XDG_CACHE_HOME only steers os.UserCacheDir on linux")
	}
	cache := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cache)

	dir, err := DefaultDir()
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(cache, "tensai", "iris")
	if dir != want {
		t.Fatalf("DefaultDir() = %q, want %q", dir, want)
	}

	// With the cache pre-populated, Load(nil) must not hit the network.
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	writeFakeIris(t, dir)
	ds, err := Load(nil)
	if err != nil {
		t.Fatal(err)
	}
	if ds.Len() != SampleCount {
		t.Fatalf("Len = %d, want %d", ds.Len(), SampleCount)
	}
}
