// Command dot prints the computation graph of a small MLP forward pass and
// its loss in Graphviz DOT format — tensai's equivalent of Gorgonia's
// encoding/dot hello world. Render it with the graphviz dot tool:
//
//	go run ./_example/dot | dot -Tsvg > graph.svg
//	go run ./_example/dot | dot -Tpng > graph.png
package main

import (
	"fmt"
	"math/rand"

	tensai "github.com/mattn/tensai"
)

func main() {
	rng := rand.New(rand.NewSource(1))

	x := tensai.Input(tensai.NewMatrix(4, 2)).Named("x")
	y := tensai.NewMatrix(4, 1)

	w1 := tensai.Param(tensai.RandomMatrix(2, 8, rng)).Named("w1")
	b1 := tensai.Param(tensai.NewMatrix(1, 8)).Named("b1")
	w2 := tensai.Param(tensai.RandomMatrix(8, 1, rng)).Named("w2")
	b2 := tensai.Param(tensai.NewMatrix(1, 1)).Named("b2")

	loss := x.MatMul(w1).AddRow(b1).Tanh().
		MatMul(w2).AddRow(b2).Sigmoid().
		MSELoss(y).Named("loss")

	fmt.Print(loss.ToDot())
}
