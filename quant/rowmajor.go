package quant

import "github.com/mattn/tensai/internal/workpool"

// PackRowMajor returns the int8 weights as little-endian uint32 words,
// four consecutive columns per word. The last word of each row is padded
// with zeros. GPU uploads use this layout instead of the CPU's row quads.
func (q *QMatrix) PackRowMajor() []uint32 {
	words := (q.Cols + 3) / 4
	out := make([]uint32, q.Rows*words)
	quads := (q.Rows + 3) / 4
	// Walk whole column tiles, so both source and destination accesses
	// have locality. The previous per-row walk jumped between large tiles
	// for every 32 bytes of output, rereading each source cache line.
	workpool.Bulk(q.Cols, 64, func(lo, hi int) {
		for col := lo; col < hi; col += q4Tile {
			cols := min(q4Tile, q.Cols-col)
			tile := q.Q[(col/q4Tile)*quads*4*q4Tile:]
			for row := 0; row < q.Rows; row += 4 {
				packRowQuad(out[row*words+col/4:], tile[(row/4)*4*q4Tile:], words, min(4, q.Rows-row), cols)
			}
		}
	})
	return out
}

func packRowQuadGeneric(dst []uint32, src []int8, stride, rows, cols int) {
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c += 4 {
			var word uint32
			for j := 0; j < min(4, cols-c); j++ {
				word |= uint32(uint8(src[(c+j)*4+r])) << (8 * j)
			}
			dst[r*stride+c/4] = word
		}
	}
}
