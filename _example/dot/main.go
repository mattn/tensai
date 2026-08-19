// Command dot prints the computation graph of z = x + y in Graphviz DOT
// format — tensai's equivalent of Gorgonia's encoding/dot example. Render
// it with the graphviz dot tool:
//
//	go run ./_example/dot | dot -Tsvg > graph.svg
//	go run ./_example/dot | dot -Tpng > graph.png
//
// ToDot works on any node, so the same call draws a full model: build a
// loss with the autograd API and print loss.ToDot() instead.
package main

import (
	"fmt"

	tensai "github.com/mattn/tensai"
)

func main() {
	x := tensai.Param(tensai.NewMatrix(1, 1)).Named("x")
	y := tensai.Param(tensai.NewMatrix(1, 1)).Named("y")
	z := x.Add(y).Named("z")

	fmt.Print(z.ToDot())
}
