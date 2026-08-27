// Package dims holds the tensor-shape arithmetic shared by the root
// package and the GPU backend: element counts, equality, and NumPy-style
// broadcasting.
package dims

import "fmt"

// Prod returns the element count of a shape; the empty shape counts as a
// single scalar element.
func Prod(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Same reports whether two shapes are identical.
func Same(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, d := range a {
		if b[i] != d {
			return false
		}
	}
	return true
}

// Broadcast combines two shapes under NumPy rules: the shapes are aligned
// at their trailing dimensions and each aligned pair must be equal or
// contain a 1.
func Broadcast(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 1; i <= n; i++ {
		da, db := 1, 1
		if i <= len(a) {
			da = a[len(a)-i]
		}
		if i <= len(b) {
			db = b[len(b)-i]
		}
		switch {
		case da == db, db == 1:
			out[n-i] = da
		case da == 1:
			out[n-i] = db
		default:
			return nil, fmt.Errorf("tensai: cannot broadcast %v with %v", a, b)
		}
	}
	return out, nil
}

// BroadcastStrides returns, for each axis of the broadcast shape out, the
// element stride to advance through a contiguous tensor of the given shape:
// 0 for axes the tensor is broadcast along (including the leading axes it
// lacks), its contiguous stride otherwise.
func BroadcastStrides(shape, out []int) []int {
	strides := make([]int, len(out))
	stride := 1
	for i := 1; i <= len(shape); i++ {
		d := len(out) - i
		if shape[len(shape)-i] == 1 && out[d] != 1 {
			strides[d] = 0
		} else {
			strides[d] = stride
		}
		stride *= shape[len(shape)-i]
	}
	return strides
}
