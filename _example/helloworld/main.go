// Command helloworld is the smallest possible tensai program: build a
// computation graph that adds two values, evaluate it, and differentiate it
// — tensai's equivalent of Gorgonia's hello world.
package main

import (
	"fmt"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/autograd"
)

func scalar(v float32) *tensai.Matrix {
	m := tensai.NewMatrix(1, 1)
	m.Data[0] = v
	return m
}

func main() {
	x := autograd.Param(scalar(2)).Named("x")
	y := autograd.Param(scalar(3)).Named("y")

	// The graph is built by simply writing the expression.
	z := x.Add(y).Named("z")
	fmt.Printf("z = %v\n", z.Scalar())

	// Reverse-mode autodiff fills in the gradients.
	z.Backward()
	fmt.Printf("dz/dx = %v\n", x.Grad().Data[0])
	fmt.Printf("dz/dy = %v\n", y.Grad().Data[0])
}
