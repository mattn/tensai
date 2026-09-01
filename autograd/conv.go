package autograd

import (
	"fmt"

	"github.com/mattn/tensai"
)

// Conv describes a square convolution window.
type Conv struct {
	Kernel int // window edge
	Stride int // step between windows, 1 when zero
	Pad    int // zeros added on each side
}

// stride returns the configured stride, defaulting to one.
func (c Conv) stride() int {
	if c.Stride == 0 {
		return 1
	}
	return c.Stride
}

// out returns the output edge for an input edge.
func (c Conv) out(in int) int {
	return (in+2*c.Pad-c.Kernel)/c.stride() + 1
}

// Im2Col expands a (batch, channels, height, width) image into the patch
// matrix a convolution multiplies: (batch, outH*outW, channels*k*k), one
// row per output pixel and one column per (channel, ky, kx) position.
// Positions the padding puts outside the image read as zero, and on the
// way back their gradients are dropped rather than scattered.
//
// It is the only piece a convolution needs that is not already an
// operation here: with it, Conv2D is an Im2Col, a MatMul, and a reshape.
func (n *Node) Im2Col(c Conv) *Node {
	shape := n.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensai: im2col needs a (batch, channels, height, width) tensor, got %s", shapeString(shape)))
	}
	batch, ch, h, w := shape[0], shape[1], shape[2], shape[3]
	if c.Kernel <= 0 {
		panic("tensai: im2col needs a positive kernel")
	}
	outH, outW := c.out(h), c.out(w)
	if outH <= 0 || outW <= 0 {
		panic(fmt.Sprintf("tensai: im2col window %d does not fit a %dx%d image", c.Kernel, h, w))
	}
	patch := ch * c.Kernel * c.Kernel
	v := n.tape.zeros([]int{batch, outH * outW, patch})
	src := n.Value().Data
	stride, pad, k := c.stride(), c.Pad, c.Kernel

	// walk visits every (output pixel, kernel position) pair that lands
	// inside the image, with the flat index of each side.
	walk := func(visit func(dst, srcIdx int)) {
		for m := 0; m < batch; m++ {
			base := m * ch * h * w
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					row := ((m*outH+oy)*outW + ox) * patch
					for cc := 0; cc < ch; cc++ {
						for ky := 0; ky < k; ky++ {
							y := oy*stride - pad + ky
							if y < 0 || y >= h {
								continue
							}
							for kx := 0; kx < k; kx++ {
								x := ox*stride - pad + kx
								if x < 0 || x >= w {
									continue
								}
								visit(row+(cc*k+ky)*k+kx, base+(cc*h+y)*w+x)
							}
						}
					}
				}
			}
		}
	}
	walk(func(dst, srcIdx int) { v.Data[dst] = src[srcIdx] })

	return newNode("im2col", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		grad := out.Grad().Data
		// A pixel appears in every window that covers it, so the patches
		// accumulate back into it.
		walk(func(dst, srcIdx int) { g.Data[srcIdx] += grad[dst] })
	})
}

// Conv2D convolves a (batch, channels, height, width) image with weights
// shaped (channels*k*k, outChannels) -- the layout Im2Col multiplies
// against -- and returns (batch, outChannels, outH, outW). bias may be nil
// or hold one value per output channel.
func (n *Node) Conv2D(w, bias *Node, c Conv) *Node {
	shape := n.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensai: conv2d needs a (batch, channels, height, width) tensor, got %s", shapeString(shape)))
	}
	batch, outH, outW := shape[0], c.out(shape[2]), c.out(shape[3])
	wShape := w.Shape()
	if len(wShape) != 2 || wShape[0] != shape[1]*c.Kernel*c.Kernel {
		panic(fmt.Sprintf("tensai: conv2d weights are %s, want (%d, outChannels)",
			shapeString(wShape), shape[1]*c.Kernel*c.Kernel))
	}
	outC := wShape[1]
	// (batch, pixels, patch) * (patch, outC) -> (batch, pixels, outC), then
	// the channel axis moves in front of the pixels and splits back into a
	// height and a width.
	out := n.Im2Col(c).MatMul(w).Transpose(0, 2, 1).Reshape(batch, outC, outH, outW)
	if bias == nil {
		return out
	}
	if len(bias.Shape()) == 1 {
		bias = bias.Reshape(outC, 1, 1)
	}
	return out.Add(bias)
}

// MaxPool2D takes the largest value of each square window of a (batch,
// channels, height, width) image, with the stride equal to the window.
// Gradients flow only to the position that won.
func (n *Node) MaxPool2D(size int) *Node {
	shape := n.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensai: maxpool needs a (batch, channels, height, width) tensor, got %s", shapeString(shape)))
	}
	if size <= 0 {
		panic("tensai: maxpool needs a positive window")
	}
	batch, ch, h, w := shape[0], shape[1], shape[2], shape[3]
	outH, outW := h/size, w/size
	if outH == 0 || outW == 0 {
		panic(fmt.Sprintf("tensai: maxpool window %d does not fit a %dx%d image", size, h, w))
	}
	v := n.tape.tensor([]int{batch, ch, outH, outW})
	// One index per output: where the gradient goes on the way back.
	from := make([]int32, len(v.Data))
	src := n.Value().Data
	for m := 0; m < batch; m++ {
		for cc := 0; cc < ch; cc++ {
			plane := (m*ch + cc) * h * w
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					best := plane + (oy*size)*w + ox*size
					for ky := 0; ky < size; ky++ {
						for kx := 0; kx < size; kx++ {
							i := plane + (oy*size+ky)*w + ox*size + kx
							if src[i] > src[best] {
								best = i
							}
						}
					}
					o := ((m*ch+cc)*outH+oy)*outW + ox
					v.Data[o] = src[best]
					from[o] = int32(best)
				}
			}
		}
	}
	return newNode("maxpool", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		for o, gv := range out.Grad().Data {
			g.Data[from[o]] += gv
		}
	})
}

// AvgPool2D averages each square window instead, spreading the gradient
// evenly over the window it came from.
func (n *Node) AvgPool2D(size int) *Node {
	shape := n.Shape()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensai: avgpool needs a (batch, channels, height, width) tensor, got %s", shapeString(shape)))
	}
	if size <= 0 {
		panic("tensai: avgpool needs a positive window")
	}
	batch, ch, h, w := shape[0], shape[1], shape[2], shape[3]
	outH, outW := h/size, w/size
	if outH == 0 || outW == 0 {
		panic(fmt.Sprintf("tensai: avgpool window %d does not fit a %dx%d image", size, h, w))
	}
	v := n.tape.zeros([]int{batch, ch, outH, outW})
	src := n.Value().Data
	inv := 1 / tensai.Float(size*size)
	for m := 0; m < batch; m++ {
		for cc := 0; cc < ch; cc++ {
			plane := (m*ch + cc) * h * w
			for oy := 0; oy < outH; oy++ {
				for ox := 0; ox < outW; ox++ {
					var sum tensai.Float
					for ky := 0; ky < size; ky++ {
						row := src[plane+(oy*size+ky)*w+ox*size:]
						for kx := 0; kx < size; kx++ {
							sum += row[kx]
						}
					}
					v.Data[((m*ch+cc)*outH+oy)*outW+ox] = sum * inv
				}
			}
		}
	}
	return newNode("avgpool", v, n).withBack(func(out *Node) {
		g := n.ensureGrad()
		grad := out.Grad().Data
		for m := 0; m < batch; m++ {
			for cc := 0; cc < ch; cc++ {
				plane := (m*ch + cc) * h * w
				for oy := 0; oy < outH; oy++ {
					for ox := 0; ox < outW; ox++ {
						gv := grad[((m*ch+cc)*outH+oy)*outW+ox] * inv
						for ky := 0; ky < size; ky++ {
							row := g.Data[plane+(oy*size+ky)*w+ox*size:]
							for kx := 0; kx < size; kx++ {
								row[kx] += gv
							}
						}
					}
				}
			}
		}
	})
}
