package tensai

import (
	"fmt"
	"math/rand"
)

// Dropout randomly zeroes elements during training with probability Rate and
// scales the survivors by 1/(1-Rate) ("inverted dropout"), so inference is a
// plain pass-through with no rescaling.
type Dropout struct {
	Rate Float

	rng  *rand.Rand
	mask []Float

	bufferPair
}

// NewDropout returns a Dropout layer that drops the given fraction of
// activations during training. Rate must be in [0, 1).
func NewDropout(rate Float) *Dropout {
	return &Dropout{Rate: rate}
}

func (d *Dropout) Init(inputCols int, rng *rand.Rand) (int, error) {
	if d.Rate < 0 || d.Rate >= 1 {
		return 0, fmt.Errorf("tensai: dropout rate must be in [0,1): %g", d.Rate)
	}
	d.rng = rng
	return inputCols, nil
}

func (d *Dropout) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if !d.training || d.Rate == 0 {
		d.mask = nil
		return input, nil
	}
	keep := 1 - d.Rate
	if cap(d.mask) < len(input.Data) {
		d.mask = make([]Float, len(input.Data))
	}
	d.mask = d.mask[:len(input.Data)]
	out := d.fwdBuf(input.Rows, input.Cols)
	for i, v := range input.Data {
		if Float(d.rng.Float64()) < keep {
			d.mask[i] = 1 / keep
		} else {
			d.mask[i] = 0
		}
		out.Data[i] = v * d.mask[i]
	}
	return out, nil
}

func (d *Dropout) Backward(gradOutput *Matrix) (*Matrix, error) {
	if d.mask == nil {
		return gradOutput, nil
	}
	if len(gradOutput.Data) != len(d.mask) {
		return nil, fmt.Errorf("tensai: dropout backward shape mismatch: grad=%d mask=%d",
			len(gradOutput.Data), len(d.mask))
	}
	out := d.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	for i, g := range gradOutput.Data {
		out.Data[i] = g * d.mask[i]
	}
	return out, nil
}

func (d *Dropout) Params() (*Matrix, []Float)       { return nil, nil }
func (d *Dropout) Grads() (*Matrix, []Float)        { return nil, nil }
func (d *Dropout) SetParams(*Matrix, []Float) error { return nil }
func (d *Dropout) setTraining(train bool)           { d.training = train }
