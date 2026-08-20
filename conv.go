package tensai

import (
	"fmt"
	"math/rand"
)

// Conv2D is a 2D convolution layer. Because the framework moves data as flat
// MxN matrices, each sample row must be laid out channel-major:
// index = (channel*height + y)*width + x. The output uses the same layout.
//
// The convolution is computed as a matrix product over an im2col expansion,
// so it reuses the tuned Dot kernel. Weights are stored as an
// (inC*kernel*kernel) x outC matrix.
type Conv2D struct {
	bufferPair
	inH, inW, inC int
	outC          int
	kernel        int
	stride        int
	pad           int

	outH, outW int

	weights *Matrix
	bias    []Float
	gradW   *Matrix
	gradB   []Float

	cols  *Matrix // im2col of the last forward input
	batch int

	// backward/forward scratch, reused across training steps
	prod, wT, gRe, dcols, gradIn *Matrix
}

// NewConv2D returns a convolution layer for inH x inW inputs with inC
// channels, producing outC channels with a square kernel.
func NewConv2D(inH, inW, inC, outC, kernel, stride, pad int) *Conv2D {
	return &Conv2D{inH: inH, inW: inW, inC: inC, outC: outC, kernel: kernel, stride: stride, pad: pad}
}

func (c *Conv2D) Init(inputCols int, rng *rand.Rand) (int, error) {
	if c.inH <= 0 || c.inW <= 0 || c.inC <= 0 || c.outC <= 0 || c.kernel <= 0 || c.stride <= 0 || c.pad < 0 {
		return 0, fmt.Errorf("tensai: conv2d invalid config: in=%dx%dx%d out=%d kernel=%d stride=%d pad=%d",
			c.inH, c.inW, c.inC, c.outC, c.kernel, c.stride, c.pad)
	}
	if inputCols != c.inH*c.inW*c.inC {
		return 0, fmt.Errorf("tensai: conv2d input cols %d != %dx%dx%d", inputCols, c.inH, c.inW, c.inC)
	}
	c.outH = (c.inH+2*c.pad-c.kernel)/c.stride + 1
	c.outW = (c.inW+2*c.pad-c.kernel)/c.stride + 1
	if c.outH <= 0 || c.outW <= 0 {
		return 0, fmt.Errorf("tensai: conv2d kernel %d does not fit input %dx%d (pad %d)",
			c.kernel, c.inH, c.inW, c.pad)
	}
	c.weights = RandomMatrix(c.inC*c.kernel*c.kernel, c.outC, rng)
	c.bias = make([]Float, c.outC)
	c.gradW = NewMatrix(c.weights.Rows, c.weights.Cols)
	c.gradB = make([]Float, c.outC)
	return c.outH * c.outW * c.outC, nil
}

// im2col expands the input batch into patch rows: one row per output pixel,
// one column per (channel, ky, kx) kernel position.
func (c *Conv2D) im2col(input *Matrix) *Matrix {
	patch := c.inC * c.kernel * c.kernel
	cols := ensureMatrix(c.cols, input.Rows*c.outH*c.outW, patch)
	for m := 0; m < input.Rows; m++ {
		sample := input.Data[m*input.Cols : (m+1)*input.Cols]
		for oy := 0; oy < c.outH; oy++ {
			for ox := 0; ox < c.outW; ox++ {
				row := cols.Data[((m*c.outH+oy)*c.outW+ox)*patch:]
				for ch := 0; ch < c.inC; ch++ {
					for ky := 0; ky < c.kernel; ky++ {
						y := oy*c.stride - c.pad + ky
						for kx := 0; kx < c.kernel; kx++ {
							x := ox*c.stride - c.pad + kx
							i := (ch*c.kernel+ky)*c.kernel + kx
							if y >= 0 && y < c.inH && x >= 0 && x < c.inW {
								row[i] = sample[(ch*c.inH+y)*c.inW+x]
							} else {
								row[i] = 0
							}
						}
					}
				}
			}
		}
	}
	return cols
}

func (c *Conv2D) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Cols != c.inH*c.inW*c.inC {
		return nil, fmt.Errorf("tensai: conv2d forward shape mismatch: input cols=%d, expected %d",
			input.Cols, c.inH*c.inW*c.inC)
	}
	c.batch = input.Rows
	c.cols = c.im2col(input)
	c.prod = ensureMatrix(c.prod, c.cols.Rows, c.outC)
	if err := DotInto(c.prod, c.cols, c.weights); err != nil {
		return nil, err
	}
	prod := c.prod
	area := c.outH * c.outW
	out := c.fwdBuf(input.Rows, c.outC*area)
	for m := 0; m < input.Rows; m++ {
		for p := 0; p < area; p++ {
			for oc := 0; oc < c.outC; oc++ {
				out.Data[m*out.Cols+oc*area+p] = prod.Data[(m*area+p)*c.outC+oc] + c.bias[oc]
			}
		}
	}
	return out, nil
}

func (c *Conv2D) Backward(gradOutput *Matrix) (*Matrix, error) {
	if c.cols == nil {
		return nil, fmt.Errorf("tensai: conv2d backward called before forward")
	}
	area := c.outH * c.outW
	if gradOutput.Rows != c.batch || gradOutput.Cols != c.outC*area {
		return nil, fmt.Errorf("tensai: conv2d backward shape mismatch: grad %dx%d, expected %dx%d",
			gradOutput.Rows, gradOutput.Cols, c.batch, c.outC*area)
	}

	// Regroup the gradient to match the im2col product layout.
	c.gRe = ensureMatrix(c.gRe, c.batch*area, c.outC)
	g := c.gRe
	for i := range c.gradB {
		c.gradB[i] = 0
	}
	for m := 0; m < c.batch; m++ {
		for p := 0; p < area; p++ {
			for oc := 0; oc < c.outC; oc++ {
				v := gradOutput.Data[m*gradOutput.Cols+oc*area+p]
				g.Data[(m*area+p)*c.outC+oc] = v
				c.gradB[oc] += v
			}
		}
	}

	c.gradW = ensureMatrix(c.gradW, c.weights.Rows, c.weights.Cols)
	if err := DotTAInto(c.gradW, c.cols, g); err != nil {
		return nil, err
	}

	c.wT = ensureMatrix(c.wT, c.weights.Cols, c.weights.Rows)
	if err := TInto(c.wT, c.weights); err != nil {
		return nil, err
	}
	c.dcols = ensureMatrix(c.dcols, g.Rows, c.weights.Rows)
	if err := DotInto(c.dcols, g, c.wT); err != nil {
		return nil, err
	}
	dcols := c.dcols

	// col2im: scatter-add patch gradients back to input positions.
	patch := c.inC * c.kernel * c.kernel
	c.gradIn = ensureMatrix(c.gradIn, c.batch, c.inH*c.inW*c.inC)
	gradInput := c.gradIn
	clear(gradInput.Data)
	for m := 0; m < c.batch; m++ {
		sample := gradInput.Data[m*gradInput.Cols : (m+1)*gradInput.Cols]
		for oy := 0; oy < c.outH; oy++ {
			for ox := 0; ox < c.outW; ox++ {
				row := dcols.Data[((m*c.outH+oy)*c.outW+ox)*patch:]
				for ch := 0; ch < c.inC; ch++ {
					for ky := 0; ky < c.kernel; ky++ {
						y := oy*c.stride - c.pad + ky
						if y < 0 || y >= c.inH {
							continue
						}
						for kx := 0; kx < c.kernel; kx++ {
							x := ox*c.stride - c.pad + kx
							if x < 0 || x >= c.inW {
								continue
							}
							sample[(ch*c.inH+y)*c.inW+x] += row[(ch*c.kernel+ky)*c.kernel+kx]
						}
					}
				}
			}
		}
	}
	return gradInput, nil
}

func (c *Conv2D) Params() (*Matrix, []Float) { return c.weights, c.bias }
func (c *Conv2D) Grads() (*Matrix, []Float)  { return c.gradW, c.gradB }

func (c *Conv2D) SetParams(weights *Matrix, bias []Float) error {
	if weights == nil || weights.Rows != c.inC*c.kernel*c.kernel || weights.Cols != c.outC || len(bias) != c.outC {
		return fmt.Errorf("tensai: conv2d SetParams mismatch")
	}
	c.weights = weights
	c.bias = bias
	return nil
}

// MaxPool2D downsamples each channel by taking the maximum over
// non-overlapping size x size windows. It expects the same channel-major
// layout as Conv2D.
type MaxPool2D struct {
	bufferPair
	inH, inW, channels int
	size               int

	outH, outW int
	argmax     []int // flat input index of each output's max, per batch element
	batch      int
	inCols     int
}

// NewMaxPool2D returns a max-pooling layer with stride equal to size.
func NewMaxPool2D(inH, inW, channels, size int) *MaxPool2D {
	return &MaxPool2D{inH: inH, inW: inW, channels: channels, size: size}
}

func (p *MaxPool2D) Init(inputCols int, _ *rand.Rand) (int, error) {
	if p.inH <= 0 || p.inW <= 0 || p.channels <= 0 || p.size <= 0 {
		return 0, fmt.Errorf("tensai: maxpool2d invalid config: in=%dx%dx%d size=%d",
			p.inH, p.inW, p.channels, p.size)
	}
	if inputCols != p.inH*p.inW*p.channels {
		return 0, fmt.Errorf("tensai: maxpool2d input cols %d != %dx%dx%d", inputCols, p.inH, p.inW, p.channels)
	}
	p.outH = p.inH / p.size
	p.outW = p.inW / p.size
	if p.outH == 0 || p.outW == 0 {
		return 0, fmt.Errorf("tensai: maxpool2d size %d larger than input %dx%d", p.size, p.inH, p.inW)
	}
	p.inCols = inputCols
	return p.outH * p.outW * p.channels, nil
}

func (p *MaxPool2D) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Cols != p.inCols {
		return nil, fmt.Errorf("tensai: maxpool2d forward shape mismatch: input cols=%d, expected %d",
			input.Cols, p.inCols)
	}
	p.batch = input.Rows
	outCols := p.outH * p.outW * p.channels
	out := p.fwdBuf(input.Rows, outCols)
	if need := input.Rows * outCols; cap(p.argmax) < need {
		p.argmax = make([]int, need)
	} else {
		p.argmax = p.argmax[:need]
	}
	for m := 0; m < input.Rows; m++ {
		sample := input.Data[m*input.Cols : (m+1)*input.Cols]
		for ch := 0; ch < p.channels; ch++ {
			for oy := 0; oy < p.outH; oy++ {
				for ox := 0; ox < p.outW; ox++ {
					bestIdx := (ch*p.inH+oy*p.size)*p.inW + ox*p.size
					best := sample[bestIdx]
					for ky := 0; ky < p.size; ky++ {
						for kx := 0; kx < p.size; kx++ {
							idx := (ch*p.inH+oy*p.size+ky)*p.inW + ox*p.size + kx
							if sample[idx] > best {
								best = sample[idx]
								bestIdx = idx
							}
						}
					}
					o := m*outCols + (ch*p.outH+oy)*p.outW + ox
					out.Data[o] = best
					p.argmax[o] = bestIdx
				}
			}
		}
	}
	return out, nil
}

func (p *MaxPool2D) Backward(gradOutput *Matrix) (*Matrix, error) {
	if p.argmax == nil {
		return nil, fmt.Errorf("tensai: maxpool2d backward called before forward")
	}
	outCols := p.outH * p.outW * p.channels
	if gradOutput.Rows != p.batch || gradOutput.Cols != outCols {
		return nil, fmt.Errorf("tensai: maxpool2d backward shape mismatch: grad %dx%d, expected %dx%d",
			gradOutput.Rows, gradOutput.Cols, p.batch, outCols)
	}
	gradInput := p.bwdBuf(p.batch, p.inCols)
	clear(gradInput.Data)
	for m := 0; m < p.batch; m++ {
		for o := 0; o < outCols; o++ {
			gradInput.Data[m*p.inCols+p.argmax[m*outCols+o]] += gradOutput.Data[m*outCols+o]
		}
	}
	return gradInput, nil
}

func (p *MaxPool2D) Params() (*Matrix, []Float)       { return nil, nil }
func (p *MaxPool2D) Grads() (*Matrix, []Float)        { return nil, nil }
func (p *MaxPool2D) SetParams(*Matrix, []Float) error { return nil }
