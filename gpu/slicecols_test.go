//go:build (wgpu && !wgpu24 && (linux || darwin || windows)) || (wgpu24 && (linux || darwin || windows))

package gpu

import (
	"testing"

	"github.com/mattn/tensai"
)

// SliceCols must lift exactly the window asked for out of every row, and
// refuse the shapes it cannot describe.
func TestSliceCols(t *testing.T) {
	g := openTestGPU(t)
	rows, stride := 5, 12
	data := make([]float32, rows*stride)
	for i := range data {
		data[i] = float32(i)
	}
	src, err := g.Upload(&tensai.Tensor{Shape: []int{rows, stride}, Data: data})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Free()

	for _, tt := range []struct{ off, cols int }{{0, 4}, {4, 3}, {7, 5}, {0, 12}, {11, 1}} {
		got, err := src.SliceCols(tt.off, tt.cols)
		if err != nil {
			t.Fatalf("SliceCols(%d,%d): %v", tt.off, tt.cols, err)
		}
		out, err := got.Download()
		got.Free()
		if err != nil {
			t.Fatal(err)
		}
		if s := out.Shape; len(s) != 2 || s[0] != rows || s[1] != tt.cols {
			t.Fatalf("SliceCols(%d,%d) shape %v", tt.off, tt.cols, s)
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < tt.cols; c++ {
				want := float32(r*stride + tt.off + c)
				if v := out.Data[r*tt.cols+c]; v != want {
					t.Fatalf("SliceCols(%d,%d)[%d,%d] = %v, want %v", tt.off, tt.cols, r, c, v, want)
				}
			}
		}
	}
	for _, tt := range []struct{ off, cols int }{{0, 13}, {10, 3}, {-1, 2}, {0, 0}} {
		if _, err := src.SliceCols(tt.off, tt.cols); err == nil {
			t.Errorf("SliceCols(%d,%d) was accepted", tt.off, tt.cols)
		}
	}
}
