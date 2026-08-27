package tensai

import "testing"

func TestArgmaxRow(t *testing.T) {
	m, err := NewMatrixFromSlice(3, 3, []Float{
		0.1, 0.7, 0.2,
		5, -1, 3,
		2, 2, 2, // tie goes to the lowest index
	})
	if err != nil {
		t.Fatal(err)
	}
	for r, want := range []int{1, 0, 0} {
		if got := m.ArgmaxRow(r); got != want {
			t.Fatalf("ArgmaxRow(%d) = %d, want %d", r, got, want)
		}
	}
}
