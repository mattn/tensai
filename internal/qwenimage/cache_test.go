package qwenimage

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestCacheRoundTrip quantizes a slice of the transformer, writes a
// cache, reads it back and checks the weights came through unchanged.
// It writes into a temporary copy of the checkpoint's index rather than
// the checkpoint itself, so a run leaves nothing behind.
func TestCacheRoundTrip(t *testing.T) {
	for _, bits := range []int{8, 4} {
		t.Run(fmt.Sprintf("int%d", bits), func(t *testing.T) { cacheRoundTrip(t, bits) })
	}
}

func cacheRoundTrip(t *testing.T, bits int) {
	dir := transformerDir(t)
	const layers = 1
	src, err := loadTransformer(dir, bits, layers)
	if err != nil {
		t.Fatal(err)
	}
	// Stand the cache up in a directory of its own, stamped by files of
	// its own: what the stamp is over does not matter here, only that
	// the writer and the reader see the same thing, and standing in for
	// the checkpoint keeps the test from touching it.
	tmp := t.TempDir()
	writeStandIns(t, tmp)
	if err := writeCache(tmp, bits, src.walk); err != nil {
		t.Fatal(err)
	}

	got := &Transformer{blocks: []*Block{{}}}
	release, err := readCache(tmp, bits, got.walk)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	same := func(name string, a, b *linear) {
		t.Helper()
		if a.rot != b.rot {
			t.Fatalf("%s: rotation width %d became %d", name, a.rot, b.rot)
		}
		if (a.q == nil) != (b.q == nil) || (a.q4 == nil) != (b.q4 == nil) {
			t.Fatalf("%s: the two sides are stored differently", name)
		}
		if a.q4 != nil {
			if a.q4.Rows != b.q4.Rows || a.q4.Cols != b.q4.Cols || a.q4.Group != b.q4.Group {
				t.Fatalf("%s: %dx%d group %d against %dx%d group %d", name,
					a.q4.Rows, a.q4.Cols, a.q4.Group, b.q4.Rows, b.q4.Cols, b.q4.Group)
			}
			for i := range a.q4.Q {
				if a.q4.Q[i] != b.q4.Q[i] {
					t.Fatalf("%s: nibble pair %d is %d, wrote %d", name, i, b.q4.Q[i], a.q4.Q[i])
				}
			}
			for i := range a.q4.Scale {
				if a.q4.Scale[i] != b.q4.Scale[i] {
					t.Fatalf("%s: scale %d is %v, wrote %v", name, i, b.q4.Scale[i], a.q4.Scale[i])
				}
			}
			for i := range a.q4.ScaleMin {
				if a.q4.ScaleMin[i] != b.q4.ScaleMin[i] {
					t.Fatalf("%s: packed scale %d is %d, wrote %d", name, i, b.q4.ScaleMin[i], a.q4.ScaleMin[i])
				}
			}
			return
		}
		if a.q != nil {
			if a.q.Rows != b.q.Rows || a.q.Cols != b.q.Cols {
				t.Fatalf("%s: %dx%d against %dx%d", name, a.q.Rows, a.q.Cols, b.q.Rows, b.q.Cols)
			}
			for i := range a.q.Q {
				if a.q.Q[i] != b.q.Q[i] {
					t.Fatalf("%s: weight %d is %d, wrote %d", name, i, b.q.Q[i], a.q.Q[i])
				}
			}
			for i := range a.q.Scale {
				if a.q.Scale[i] != b.q.Scale[i] {
					t.Fatalf("%s: scale %d is %v, wrote %v", name, i, b.q.Scale[i], a.q.Scale[i])
				}
			}
			return
		}
		for i := range a.f.Data {
			if a.f.Data[i] != b.f.Data[i] {
				t.Fatalf("%s: element %d is %v, wrote %v", name, i, b.f.Data[i], a.f.Data[i])
			}
		}
	}
	same("img_in", src.imgIn, got.imgIn)
	same("modulation", src.modulation, got.modulation)
	same("block 0 to_q", src.blocks[0].toQ, got.blocks[0].toQ)
	same("block 0 mlp.out", src.blocks[0].mlpOut, got.blocks[0].mlpOut)
	for i := range src.txtNorm {
		if src.txtNorm[i] != got.txtNorm[i] {
			t.Fatalf("txt_norm %d is %v, wrote %v", i, got.txtNorm[i], src.txtNorm[i])
		}
	}

	// A checkpoint that has moved on must not be read from an old cache.
	touch(t, tmp)
	if _, err := readCache(tmp, bits, (&Transformer{blocks: []*Block{{}}}).walk); err == nil {
		t.Error("a cache older than its checkpoint was accepted")
	}
}

// writeStandIns puts a couple of files where the stamp looks for
// weights, so the round trip has something to be tied to.
func writeStandIns(t *testing.T, dir string) {
	t.Helper()
	for _, name := range []string{"a.safetensors", "b.safetensors"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func filepathExt(name string) string { return filepath.Ext(name) }

// touch moves every weight's timestamp on, which is what a rebuilt or
// redownloaded checkpoint looks like.
func touch(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if filepathExt(e.Name()) != ".safetensors" {
			continue
		}
		info, err := e.Info()
		if err != nil {
			t.Fatal(err)
		}
		later := info.ModTime().Add(1e9)
		if err := os.Chtimes(filepath.Join(dir, e.Name()), later, later); err != nil {
			t.Skipf("cannot move the checkpoint's timestamp: %v", err)
		}
	}
}
