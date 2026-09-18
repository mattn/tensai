package llm

// The width the decode weights take when nobody chose one. A file's own
// quantization sets the ceiling: int4 blocks gain nothing from int8 and
// repack exactly into int4, while everything wider keeps its precision
// only at int8. Above that the machine decides, since a model that does
// not fit at int8 is faster and better at int4 than swapping.

import (
	"fmt"
	"strings"

	"github.com/mattn/tensai/encoding/gguf"
	"github.com/mattn/tensai/internal/sysmem"
)

// BitsAuto asks Open to choose the width from the file and the machine.
const BitsAuto = -1

// pickBits chooses a width for params weights stored as stored (a gguf
// type name, or "" for a float checkpoint) given avail bytes of memory,
// zero when unknown, and says why.
func pickBits(stored string, params int64, avail int64) (int, string) {
	switch stored {
	case "Q4_0", "Q4_1", "Q4_K", "IQ4_NL", "IQ4_XS", "MXFP4":
		return 4, stored + " blocks repack into int4 as they are"
	case "PTQ1_0", "PQ2_0":
		return 8, stored + " blocks are ternary whatever the width"
	}
	// int8 wants the weights plus a quarter for scales and sums, and the
	// context and the activations on top; half a gigabyte covers a
	// modest one, and the estimate leans toward not swapping.
	need8 := params*5/4 + 512<<20
	if avail == 0 {
		return 8, "int8 keeps the stored precision; memory unknown"
	}
	if need8 <= avail {
		return 8, fmt.Sprintf("int8 keeps the stored precision and %.1fGB fits the %.1fGB available",
			float64(need8)/1e9, float64(avail)/1e9)
	}
	return 4, fmt.Sprintf("int8 would want %.1fGB of the %.1fGB available, so int4",
		float64(need8)/1e9, float64(avail)/1e9)
}

// ggufBits chooses a width for an open gguf: the stored type that holds
// most of the layer weights, and the weight count from the tensor
// directory.
func ggufBits(g *gguf.File) (int, string) {
	byType := map[string]int64{}
	var params int64
	for _, name := range g.Names() {
		typ, shape, _ := g.Info(name)
		n := int64(1)
		for _, d := range shape {
			n *= int64(d)
		}
		params += n
		if strings.HasPrefix(name, "blk.") {
			byType[typ] += n
		}
	}
	stored := ""
	for typ, n := range byType {
		if stored == "" || n > byType[stored] {
			stored = typ
		}
	}
	return pickBits(stored, params, sysmem.Available())
}
