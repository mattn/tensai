package qwenimage

import (
	"math"

	"github.com/mattn/tensai"
)

// A forward pass runs over one joint sequence: the prompt's tokens, then
// the target image's latent grid laid out row by row. Three things vary
// along it and are decided here rather than in the blocks — where each
// token sits on the three rotary axes, which keys it may read, and which
// row of the modulation it takes.
//
// Text advances a shared position on all three axes, one step per token.
// The image freezes the frame axis at the position the text reached and
// spreads its tokens over a height and width grid centred on zero, so
// the same picture rotates the same way however long the prompt was.
//
// Attention is block-causal: a text token reads itself and what came
// before it, and every image token reads the whole sequence. That is
// what lets the prompt's keys and values be computed once and reused
// for every step of the denoising loop, since they also modulate from a
// timestep of zero and so never change.

// ropeTheta is the base the rotary frequencies come from.
const ropeTheta = 10000

// ropeAxes are the head dimensions the frame, height and width axes
// take; half of each is a rotating pair.
var ropeAxes = [3]int{16, 56, 56}

// Layout describes one joint sequence.
type Layout struct {
	TextLen       int // prompt tokens, which come first
	Height, Width int // the target image's latent grid

	// KeyLimit is one past the last key each query may read.
	KeyLimit []int
	// Row is the modulation row each token takes: 0 for the sampled
	// timestep, which only the target image uses, and 1 for t=0.
	Row []int
}

// NewLayout describes a text-to-image sequence: textLen prompt tokens
// followed by a height by width latent grid.
func NewLayout(textLen, height, width int) *Layout {
	n := textLen + height*width
	l := &Layout{
		TextLen:  textLen,
		Height:   height,
		Width:    width,
		KeyLimit: make([]int, n),
		Row:      make([]int, n),
	}
	for i := 0; i < textLen; i++ {
		l.KeyLimit[i] = i + 1
		l.Row[i] = 1
	}
	for i := textLen; i < n; i++ {
		l.KeyLimit[i] = n
	}
	return l
}

// Tokens is the length of the joint sequence.
func (l *Layout) Tokens() int { return len(l.Row) }

// positions returns each token's position on the three rotary axes.
func (l *Layout) positions() [3][]int {
	n := l.Tokens()
	var pos [3][]int
	for a := range pos {
		pos[a] = make([]int, n)
	}
	for i := 0; i < l.TextLen; i++ {
		pos[0][i], pos[1][i], pos[2][i] = i, i, i
	}
	// The grid is centred, so an odd extent keeps the extra row or
	// column on the negative side, as the reference's halving does.
	h0, w0 := -(l.Height - l.Height/2), -(l.Width - l.Width/2)
	for p := 0; p < l.Height*l.Width; p++ {
		i := l.TextLen + p
		pos[0][i] = l.TextLen
		pos[1][i] = h0 + p/l.Width
		pos[2][i] = w0 + p%l.Width
	}
	return pos
}

// Rope builds the rotation every token's head pairs take. The three
// axes' pairs sit side by side in the head: eight for the frame, then
// twenty-eight each for height and width.
func (l *Layout) Rope() *Rope {
	n := l.Tokens()
	r := &Rope{
		Cos: tensai.NewMatrix(n, ropePairs),
		Sin: tensai.NewMatrix(n, ropePairs),
	}
	pos := l.positions()
	base := 0
	for a, dim := range ropeAxes {
		for j := 0; j < dim/2; j++ {
			inv := math.Pow(ropeTheta, -2*float64(j)/float64(dim))
			for i := 0; i < n; i++ {
				angle := float64(pos[a][i]) * inv
				r.Cos.Data[i*ropePairs+base+j] = tensai.Float(math.Cos(angle))
				r.Sin.Data[i*ropePairs+base+j] = tensai.Float(math.Sin(angle))
			}
		}
		base += dim / 2
	}
	return r
}
