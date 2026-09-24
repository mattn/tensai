package qwenimage

import (
	"fmt"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
)

func (c *conv) onGPU(g *gpu.Device, x *tensai.Matrix, h, w int) (*tensai.Matrix, error) {
	limit := min(uint64(imcolBudget), g.StorageLimit())
	if g.StorageLimit() < uint64(c.w.Cols*c.w.Rows)*4 {
		return nil, fmt.Errorf("qwenimage: decoder filter exceeds GPU storage limit")
	}
	weight, err := g.Upload(c.w.Tensor())
	if err != nil {
		return nil, err
	}
	defer weight.Free()
	out := tensai.NewMatrix(h*w, c.out)
	tile := int(limit / (uint64(max(c.w.Cols, c.out)) * 4))
	if tile == 0 {
		return nil, fmt.Errorf("qwenimage: GPU cannot hold a decoder row")
	}
	var buf *tensai.Matrix
	if c.ksz != 1 {
		buf = tensai.NewMatrix(min(tile, h*w), c.w.Cols)
	}
	for lo := 0; lo < h*w; lo += tile {
		hi := min(lo+tile, h*w)
		var in *tensai.Matrix
		if c.ksz == 1 {
			in = &tensai.Matrix{Rows: hi - lo, Cols: c.in, Data: x.Data[lo*c.in : hi*c.in]}
		} else {
			buf.Rows = hi - lo
			buf.Data = buf.Data[:buf.Rows*buf.Cols]
			c.imcol(buf, x, h, w, lo, hi)
			in = buf
		}
		if err := decoderProduct(g, weight, in, out.Data[lo*c.out:hi*c.out]); err != nil {
			return nil, err
		}
	}
	addBias(out, c.b)
	return out, nil
}

func decoderProduct(g *gpu.Device, weight *gpu.Tensor, x *tensai.Matrix, dst []tensai.Float) error {
	in, err := g.Upload(x.Tensor())
	if err != nil {
		return err
	}
	defer in.Free()
	out, err := in.MatMulT(weight)
	if err != nil {
		return err
	}
	defer out.Free()
	host, err := out.Download()
	if err != nil {
		return err
	}
	copy(dst, host.Data)
	return nil
}
