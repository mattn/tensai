package mnist

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// writeIDX writes a gzipped IDX file so tests never touch the network.
func writeIDX(t *testing.T, dir, name string, header []uint32, body []byte) {
	t.Helper()
	var buf bytes.Buffer
	for _, v := range header {
		if err := binary.Write(&buf, binary.BigEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	buf.Write(body)
	var gzBuf bytes.Buffer
	gz := gzip.NewWriter(&gzBuf)
	if _, err := gz.Write(buf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name+".gz"), gzBuf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
}

// writeFakeMNIST fills dir with a tiny fake dataset: trainN training and
// testN test images whose first pixel is 255 and whose label is row%10.
func writeFakeMNIST(t *testing.T, dir string, trainN, testN int) {
	t.Helper()
	write := func(imageFile, labelFile string, n int) {
		images := make([]byte, n*ImageSize)
		labels := make([]byte, n)
		for i := 0; i < n; i++ {
			images[i*ImageSize] = 255
			labels[i] = byte(i % 10)
		}
		writeIDX(t, dir, imageFile, []uint32{imageMagic, uint32(n), ImageHeight, ImageWidth}, images)
		writeIDX(t, dir, labelFile, []uint32{labelMagic, uint32(n)}, labels)
	}
	write(files[0], files[1], trainN)
	write(files[2], files[3], testN)
}

func TestLoadFromDir(t *testing.T) {
	dir := t.TempDir()
	writeFakeMNIST(t, dir, 3, 2)

	data, err := Load(&Options{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if data.Train.Len() != 3 || data.Test.Len() != 2 {
		t.Fatalf("train=%d test=%d, want 3 and 2", data.Train.Len(), data.Test.Len())
	}
	if cols := data.Train.Inputs.Cols; cols != ImageSize {
		t.Fatalf("train inputs cols = %d, want %d", cols, ImageSize)
	}
	if got := data.Train.Inputs.At(0, 0); got != 1 {
		t.Fatalf("first pixel = %g, want 1 (255/255)", got)
	}
	if got := data.Train.Inputs.At(0, 1); got != 0 {
		t.Fatalf("second pixel = %g, want 0", got)
	}
	for i := 0; i < 3; i++ {
		if got := data.Train.Targets.At(i, 0); int(got) != i%10 {
			t.Fatalf("train label %d = %g, want %d", i, got, i%10)
		}
	}
}

func TestLoadLimits(t *testing.T) {
	dir := t.TempDir()
	writeFakeMNIST(t, dir, 5, 4)

	data, err := Load(&Options{Dir: dir, TrainLimit: 2, TestLimit: 3})
	if err != nil {
		t.Fatal(err)
	}
	if data.Train.Len() != 2 || data.Test.Len() != 3 {
		t.Fatalf("train=%d test=%d, want 2 and 3", data.Train.Len(), data.Test.Len())
	}
	// A limit beyond the file's row count clamps to what is available.
	data, err = Load(&Options{Dir: dir, TrainLimit: 100, TestLimit: 100})
	if err != nil {
		t.Fatal(err)
	}
	if data.Train.Len() != 5 || data.Test.Len() != 4 {
		t.Fatalf("train=%d test=%d, want 5 and 4", data.Train.Len(), data.Test.Len())
	}
}

func TestLoadBadMagic(t *testing.T) {
	dir := t.TempDir()
	writeFakeMNIST(t, dir, 1, 1)
	writeIDX(t, dir, files[0], []uint32{1234, 1, ImageHeight, ImageWidth}, make([]byte, ImageSize))

	if _, err := Load(&Options{Dir: dir}); err == nil || !strings.Contains(err.Error(), "bad image header") {
		t.Fatalf("err = %v, want bad image header", err)
	}
}

func TestLoadBadLabel(t *testing.T) {
	dir := t.TempDir()
	writeFakeMNIST(t, dir, 1, 1)
	writeIDX(t, dir, files[1], []uint32{labelMagic, 1}, []byte{200})

	if _, err := Load(&Options{Dir: dir}); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("err = %v, want label out of range", err)
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
	want := filepath.Join(cache, "tensai", "mnist")
	if dir != want {
		t.Fatalf("DefaultDir() = %q, want %q", dir, want)
	}

	// With the cache pre-populated, Load(nil) must not hit the network.
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	writeFakeMNIST(t, dir, 2, 1)
	data, err := Load(nil)
	if err != nil {
		t.Fatal(err)
	}
	if data.Train.Len() != 2 || data.Test.Len() != 1 {
		t.Fatalf("train=%d test=%d, want 2 and 1", data.Train.Len(), data.Test.Len())
	}
}
