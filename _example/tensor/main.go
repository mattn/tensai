// Command tensor tours the n-dimensional Tensor API: NumPy-style
// broadcasting, batched matrix multiplication with a shared weight, and
// scaled dot-product attention over a whole batch in three lines.
package main

import (
	"fmt"
	"math"
	"math/rand"

	tensai "github.com/mattn/tensai"
)

func randTensor(rng *rand.Rand, shape ...int) *tensai.Tensor {
	t := tensai.NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = float32(rng.NormFloat64())
	}
	return t
}

// softmaxLast applies softmax along the last axis, in place. Tensors are
// contiguous and row-major, so the last axis is a series of plain slices.
func softmaxLast(t *tensai.Tensor) {
	n := t.Shape[len(t.Shape)-1]
	for pos := 0; pos < len(t.Data); pos += n {
		row := t.Data[pos : pos+n]
		maxv := row[0]
		for _, v := range row[1:] {
			if v > maxv {
				maxv = v
			}
		}
		var sum float32
		for i, v := range row {
			e := float32(math.Exp(float64(v - maxv)))
			row[i] = e
			sum += e
		}
		for i := range row {
			row[i] /= sum
		}
	}
}

func main() {
	rng := rand.New(rand.NewSource(42))

	// Broadcasting: center a batch of sequences by per-channel means. The
	// (channel) vector stretches across both the batch and position axes.
	x := randTensor(rng, 4, 6, 3) // (batch, position, channel)
	mean, err := tensai.NewTensorFromSlice([]float32{0.5, -1, 2}, 3)
	if err != nil {
		panic(err)
	}
	centered, err := x.Sub(mean)
	if err != nil {
		panic(err)
	}
	fmt.Printf("x%v - mean%v = centered%v\n", x.Shape, mean.Shape, centered.Shape)

	// Batched MatMul with a shared weight: the 2-D matrix broadcasts
	// across the batch axis, projecting every sequence in one call.
	w := randTensor(rng, 3, 8)
	h, err := tensai.MatMul(centered, w)
	if err != nil {
		panic(err)
	}
	fmt.Printf("centered%v @ w%v = h%v\n", centered.Shape, w.Shape, h.Shape)

	// Scaled dot-product attention for the whole batch: Transpose swaps
	// the last two axes, so q @ k^T and the value mix are each one MatMul.
	q, k, v := h, randTensor(rng, 4, 6, 8), randTensor(rng, 4, 6, 8)
	kt, err := k.Transpose()
	if err != nil {
		panic(err)
	}
	scores, err := tensai.MatMul(q, kt)
	if err != nil {
		panic(err)
	}
	scores.Scale(1 / float32(math.Sqrt(8)))
	softmaxLast(scores)
	out, err := tensai.MatMul(scores, v)
	if err != nil {
		panic(err)
	}
	fmt.Printf("attention: q%v @ k^T%v -> weights%v, @ v%v = out%v\n",
		q.Shape, kt.Shape, scores.Shape, v.Shape, out.Shape)

	// Each attention row is a probability distribution over positions.
	fmt.Println("\nattention weights for batch 0, query 0:")
	var sum float32
	for j := 0; j < scores.Shape[2]; j++ {
		w := scores.At(0, 0, j)
		sum += w
		fmt.Printf("  position %d: %.4f\n", j, w)
	}
	fmt.Printf("  sum: %.4f\n", sum)
}
