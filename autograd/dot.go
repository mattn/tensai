package autograd

import (
	"fmt"
	"strings"
)

// ToDot renders the computation graph rooted at n in Graphviz DOT format,
// in the spirit of Gorgonia's encoding/dot. Pipe it through the dot tool to
// get an image:
//
//	go run ./_example/dot | dot -Tsvg > graph.svg
//
// Leaves are drawn as boxes (Param blue, Input gray) and operations as
// rounded nodes; every node shows its shape. Use Named to label leaves.
func (n *Node) ToDot() string {
	var sb strings.Builder
	sb.WriteString("digraph tensai {\n")
	sb.WriteString("  rankdir=TB;\n")
	sb.WriteString("  node [fontname=\"Helvetica\", fontsize=11];\n")
	sb.WriteString("  edge [fontname=\"Helvetica\", fontsize=9, color=\"#8A93A5\"];\n")

	ids := map[*Node]int{}
	var visit func(*Node) int
	visit = func(x *Node) int {
		if id, ok := ids[x]; ok {
			return id
		}
		id := len(ids)
		ids[x] = id

		label := x.name
		var attrs string
		switch {
		case x.op != "":
			if label == "" {
				label = x.op
			} else {
				label += " = " + x.op
			}
			attrs = `shape=box, style="rounded,filled", fillcolor="#EEF2FF"`
		case x.requiresGrad:
			if label == "" {
				label = "param"
			}
			attrs = `shape=box, style=filled, fillcolor="#DBEEFF"`
		default:
			if label == "" {
				label = "input"
			}
			attrs = `shape=box, style=filled, fillcolor="#EFEFEF"`
		}
		label += fmt.Sprintf("\\n%dx%d", x.Value.Rows, x.Value.Cols)
		fmt.Fprintf(&sb, "  n%d [label=\"%s\", %s];\n", id, label, attrs)

		for _, p := range x.parents {
			pid := visit(p)
			fmt.Fprintf(&sb, "  n%d -> n%d;\n", pid, id)
		}
		return id
	}
	visit(n)
	sb.WriteString("}\n")
	return sb.String()
}
