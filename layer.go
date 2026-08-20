package tensai

import (
	"fmt"
	"math"
	"math/rand"
)

// Layer is a single differentiable stage of a Sequential model.
// Forward and Backward are batched: inputs/outputs are MxN matrices
// where M is the batch size and N is the feature dimension.
type Layer interface {
	// Init configures parameters using the given RNG and input width.
	Init(inputCols int, rng *rand.Rand) (outputCols int, err error)

	// Forward computes activations given the input batch.
	Forward(input *Matrix) (*Matrix, error)

	// Backward computes the gradient with respect to the layer input,
	// given the gradient with respect to the layer output.
	Backward(gradOutput *Matrix) (*Matrix, error)

	// Grads returns the parameter gradients accumulated during the last
	// backward pass, in the order [weights, bias]. Layers without
	// parameters return nil.
	Grads() (*Matrix, []Float)

	// Params returns the current parameters [weights, bias].
	Params() (*Matrix, []Float)

	// SetParams replaces the parameters [weights, bias].
	SetParams(weights *Matrix, bias []Float) error
}

// Dense is a fully-connected layer: y = x*W + b.
type Dense struct {
	bufferPair
	rng *rand.Rand

	weights *Matrix // (in x out)
	bias    []Float

	gradW *Matrix
	gradB []Float

	input *Matrix
	tIn   *Matrix // scratch: input^T
	tW    *Matrix // scratch: weights^T
}

// NewDense returns a Dense layer with the given output size.
func NewDense(outCols int) *Dense {
	return &Dense{bias: make([]Float, outCols)}
}

func (d *Dense) Init(inputCols int, rng *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: dense init with non-positive input cols: %d", inputCols)
	}
	d.rng = rng
	d.weights = RandomMatrix(inputCols, d.outCols(), rng)
	d.bias = make([]Float, d.outCols())
	d.gradW = NewMatrix(inputCols, d.outCols())
	d.gradB = make([]Float, d.outCols())
	return d.outCols(), nil
}

func (d *Dense) outCols() int {
	if d.weights != nil {
		return d.weights.Cols
	}
	// Before Init, bias length is the configured output width.
	return len(d.bias)
}

func (d *Dense) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Cols != d.weights.Rows {
		return nil, fmt.Errorf("tensai: dense forward shape mismatch: input %dx%d, weights %dx%d",
			input.Rows, input.Cols, d.weights.Rows, d.weights.Cols)
	}
	d.input = input
	out := d.fwdBuf(input.Rows, d.weights.Cols)
	if err := DotInto(out, input, d.weights); err != nil {
		return nil, err
	}
	for r := 0; r < out.Rows; r++ {
		addSlice(out.Data[r*out.Cols:(r+1)*out.Cols], d.bias)
	}
	return out, nil
}

func (d *Dense) Backward(gradOutput *Matrix) (*Matrix, error) {
	if d.input == nil {
		return nil, fmt.Errorf("tensai: dense backward called before forward")
	}
	// gradW = input^T * gradOutput  (in x out)
	d.tIn = ensureMatrix(d.tIn, d.input.Cols, d.input.Rows)
	if err := TInto(d.tIn, d.input); err != nil {
		return nil, err
	}
	d.gradW = ensureMatrix(d.gradW, d.weights.Rows, d.weights.Cols)
	if err := DotInto(d.gradW, d.tIn, gradOutput); err != nil {
		return nil, err
	}

	// gradB = sum over batch of gradOutput
	clear(d.gradB)
	for r := 0; r < gradOutput.Rows; r++ {
		addSlice(d.gradB, gradOutput.Data[r*gradOutput.Cols:(r+1)*gradOutput.Cols])
	}

	// gradInput = gradOutput * weights^T  (batch x in)
	d.tW = ensureMatrix(d.tW, d.weights.Cols, d.weights.Rows)
	if err := TInto(d.tW, d.weights); err != nil {
		return nil, err
	}
	gradInput := d.bwdBuf(gradOutput.Rows, d.weights.Rows)
	if err := DotInto(gradInput, gradOutput, d.tW); err != nil {
		return nil, err
	}
	return gradInput, nil
}

func (d *Dense) Grads() (*Matrix, []Float) {
	return d.gradW, d.gradB
}

func (d *Dense) Params() (*Matrix, []Float) {
	return d.weights, d.bias
}

func (d *Dense) SetParams(weights *Matrix, bias []Float) error {
	if weights == nil || len(bias) != weights.Cols {
		return fmt.Errorf("tensai: dense SetParams mismatch: bias len=%d weights cols=%d", len(bias), weights.Cols)
	}
	d.weights = weights
	d.bias = bias
	return nil
}

// Activation is a shared interface for element-wise non-linearities.
type activation interface {
	Layer
	apply(x Float) Float
	applyGrad(x Float) Float
}

// bufferPair provides scratch matrices reused across training steps.
// Forward buffers are only reused while training (FitStep switches the mode
// on): Predict results stay valid until the next Predict of the same shape
// would otherwise silently overwrite them. Backward runs only during
// training, so its buffer is always reused.
type bufferPair struct {
	training bool
	fwd, bwd *Matrix
}

func (b *bufferPair) setTraining(v bool) { b.training = v }

func (b *bufferPair) fwdBuf(rows, cols int) *Matrix {
	if !b.training {
		return NewMatrix(rows, cols)
	}
	b.fwd = ensureMatrix(b.fwd, rows, cols)
	return b.fwd
}

func (b *bufferPair) bwdBuf(rows, cols int) *Matrix {
	b.bwd = ensureMatrix(b.bwd, rows, cols)
	return b.bwd
}

// activationBase stores the last forward input/output for reuse in backward.
type activationBase struct {
	bufferPair
	input  *Matrix
	output *Matrix
}

func (a *activationBase) Params() (*Matrix, []Float)       { return nil, nil }
func (a *activationBase) SetParams(*Matrix, []Float) error { return nil }
func (a *activationBase) Grads() (*Matrix, []Float)        { return nil, nil }

// ReLU activation: f(x) = max(0, x).
type ReLU struct{ activationBase }

func (r *ReLU) apply(x Float) Float {
	if x > 0 {
		return x
	}
	return 0
}
func (r *ReLU) applyGrad(x Float) Float {
	if x > 0 {
		return 1
	}
	return 0
}

func (r *ReLU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (r *ReLU) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	r.input = input
	out := r.fwdBuf(input.Rows, input.Cols)
	reluFwd(out.Data, input.Data)
	r.output = out
	return out, nil
}

func (r *ReLU) Backward(gradOutput *Matrix) (*Matrix, error) {
	if r.input == nil {
		return nil, fmt.Errorf("tensai: relu backward called before forward")
	}
	out := r.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	reluBwd(out.Data, gradOutput.Data, r.input.Data)
	return out, nil
}

// LeakyReLU activation: f(x) = x for x > 0, alpha*x otherwise.
type LeakyReLU struct {
	activationBase
	Alpha Float
}

// NewLeakyReLU returns a LeakyReLU with the given negative-side slope.
func NewLeakyReLU(alpha Float) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

func (l *LeakyReLU) apply(x Float) Float {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}
func (l *LeakyReLU) applyGrad(x Float) Float {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

func (l *LeakyReLU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (l *LeakyReLU) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	l.input = input
	out := l.fwdBuf(input.Rows, input.Cols)
	leakyFwd(out.Data, input.Data, l.Alpha)
	l.output = out
	return out, nil
}

func (l *LeakyReLU) Backward(gradOutput *Matrix) (*Matrix, error) {
	if l.input == nil {
		return nil, fmt.Errorf("tensai: leakyrelu backward called before forward")
	}
	out := l.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	leakyBwd(out.Data, gradOutput.Data, l.input.Data, l.Alpha)
	return out, nil
}

// Sigmoid activation: f(x) = 1 / (1 + e^-x).
type Sigmoid struct{ activationBase }

func (s *Sigmoid) apply(x Float) Float {
	return 1.0 / (1.0 + expF(-x))
}
func (s *Sigmoid) applyGrad(x Float) Float {
	// derivative in terms of the output y = sigmoid(x): y*(1-y)
	y := s.apply(x)
	return y * (1 - y)
}

func (s *Sigmoid) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (s *Sigmoid) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	s.input = input
	out := s.fwdBuf(input.Rows, input.Cols)
	sigmoidFwd(out.Data, input.Data)
	s.output = out
	return out, nil
}

func (s *Sigmoid) Backward(gradOutput *Matrix) (*Matrix, error) {
	if s.output == nil {
		return nil, fmt.Errorf("tensai: sigmoid backward called before forward")
	}
	out := s.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	sigmoidBwd(out.Data, gradOutput.Data, s.output.Data)
	return out, nil
}

// Tanh activation: f(x) = tanh(x).
type Tanh struct{ activationBase }

func (t *Tanh) apply(x Float) Float { return tanhF(x) }
func (t *Tanh) applyGrad(x Float) Float {
	y := t.apply(x)
	return 1 - y*y
}

func (t *Tanh) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (t *Tanh) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	t.input = input
	out := t.fwdBuf(input.Rows, input.Cols)
	tanhFwd(out.Data, input.Data)
	t.output = out
	return out, nil
}

func (t *Tanh) Backward(gradOutput *Matrix) (*Matrix, error) {
	if t.output == nil {
		return nil, fmt.Errorf("tensai: tanh backward called before forward")
	}
	out := t.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	tanhBwd(out.Data, gradOutput.Data, t.output.Data)
	return out, nil
}

// GELU activation: f(x) = 0.5*x*(1+erf(x/sqrt(2))).
type GELU struct{ activationBase }

func (g *GELU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (g *GELU) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	g.input = input
	out := g.fwdBuf(input.Rows, input.Cols)
	for i, x := range input.Data {
		out.Data[i] = geluF(x)
	}
	g.output = out
	return out, nil
}

func (g *GELU) Backward(gradOutput *Matrix) (*Matrix, error) {
	if g.input == nil {
		return nil, fmt.Errorf("tensai: gelu backward called before forward")
	}
	out := g.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	for i, grad := range gradOutput.Data {
		out.Data[i] = grad * geluGrad(g.input.Data[i])
	}
	return out, nil
}

// LayerNorm normalizes each row over its feature dimension and applies a
// learnable affine transform.
type LayerNorm struct {
	bufferPair

	gamma *Matrix
	beta  []Float

	gradGamma *Matrix
	gradBeta  []Float

	normalized *Matrix
	invStd     []Float
	eps        Float
}

// NewLayerNorm returns a LayerNorm with the default epsilon.
func NewLayerNorm() *LayerNorm {
	return &LayerNorm{eps: 1e-5}
}

func (l *LayerNorm) Init(inputCols int, _ *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: layernorm init with non-positive input cols: %d", inputCols)
	}
	l.gamma = NewMatrix(1, inputCols)
	for i := range l.gamma.Data {
		l.gamma.Data[i] = 1
	}
	l.beta = make([]Float, inputCols)
	l.gradGamma = NewMatrix(1, inputCols)
	l.gradBeta = make([]Float, inputCols)
	return inputCols, nil
}

func (l *LayerNorm) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if l.gamma == nil || input.Cols != l.gamma.Cols {
		return nil, fmt.Errorf("tensai: layernorm forward shape mismatch: input %dx%d gamma %v",
			input.Rows, input.Cols, shapeString(l.gamma))
	}
	out := l.fwdBuf(input.Rows, input.Cols)
	l.normalized = ensureMatrix(l.normalized, input.Rows, input.Cols)
	if cap(l.invStd) < input.Rows {
		l.invStd = make([]Float, input.Rows)
	} else {
		l.invStd = l.invStd[:input.Rows]
	}
	colsF := Float(input.Cols)
	for r := 0; r < input.Rows; r++ {
		inRow := input.Data[r*input.Cols : (r+1)*input.Cols]
		normRow := l.normalized.Data[r*input.Cols : (r+1)*input.Cols]
		outRow := out.Data[r*input.Cols : (r+1)*input.Cols]
		var mean Float
		for _, v := range inRow {
			mean += v
		}
		mean /= colsF
		var variance Float
		for _, v := range inRow {
			d := v - mean
			variance += d * d
		}
		variance /= colsF
		invStd := 1 / sqrtF(variance+l.eps)
		l.invStd[r] = invStd
		for c, v := range inRow {
			xhat := (v - mean) * invStd
			normRow[c] = xhat
			outRow[c] = xhat*l.gamma.Data[c] + l.beta[c]
		}
	}
	return out, nil
}

func (l *LayerNorm) Backward(gradOutput *Matrix) (*Matrix, error) {
	if l.normalized == nil {
		return nil, fmt.Errorf("tensai: layernorm backward called before forward")
	}
	l.gradGamma = ensureMatrix(l.gradGamma, 1, gradOutput.Cols)
	clear(l.gradGamma.Data)
	clear(l.gradBeta)
	out := l.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	colsF := Float(gradOutput.Cols)
	for r := 0; r < gradOutput.Rows; r++ {
		gRow := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		xhatRow := l.normalized.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		outRow := out.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		var sumDXhat, sumDXhatXhat Float
		for c, g := range gRow {
			l.gradGamma.Data[c] += g * xhatRow[c]
			l.gradBeta[c] += g
			dxhat := g * l.gamma.Data[c]
			sumDXhat += dxhat
			sumDXhatXhat += dxhat * xhatRow[c]
		}
		invStd := l.invStd[r]
		for c, g := range gRow {
			dxhat := g * l.gamma.Data[c]
			outRow[c] = invStd / colsF * (colsF*dxhat - sumDXhat - xhatRow[c]*sumDXhatXhat)
		}
	}
	return out, nil
}

func (l *LayerNorm) Grads() (*Matrix, []Float) {
	return l.gradGamma, l.gradBeta
}

func (l *LayerNorm) Params() (*Matrix, []Float) {
	return l.gamma, l.beta
}

func (l *LayerNorm) SetParams(weights *Matrix, bias []Float) error {
	if weights == nil || weights.Rows != 1 || len(bias) != weights.Cols {
		return fmt.Errorf("tensai: layernorm SetParams mismatch: weights=%v bias len=%d",
			shapeString(weights), len(bias))
	}
	l.gamma = weights
	l.beta = bias
	return nil
}

// Embedding looks up a learned vector for each token id in the input row and
// concatenates the vectors across columns.
type Embedding struct {
	bufferPair

	weights *Matrix
	gradW   *Matrix
	input   *Matrix
}

// NewEmbedding returns a trainable embedding table of shape vocabSize x dim.
func NewEmbedding(vocabSize, dim int) *Embedding {
	return &Embedding{weights: NewMatrix(vocabSize, dim)}
}

func (e *Embedding) Init(inputCols int, rng *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: embedding init with non-positive input cols: %d", inputCols)
	}
	if e.weights == nil || e.weights.Rows <= 0 || e.weights.Cols <= 0 {
		return 0, fmt.Errorf("tensai: embedding init with invalid table shape")
	}
	e.weights = RandomMatrix(e.weights.Rows, e.weights.Cols, rng)
	e.gradW = NewMatrix(e.weights.Rows, e.weights.Cols)
	return inputCols * e.weights.Cols, nil
}

func (e *Embedding) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	e.input = input
	outCols := input.Cols * e.weights.Cols
	out := e.fwdBuf(input.Rows, outCols)
	for r := 0; r < input.Rows; r++ {
		outRow := out.Data[r*out.Cols : (r+1)*out.Cols]
		for c := 0; c < input.Cols; c++ {
			idx, err := embeddingIndex(input.At(r, c), e.weights.Rows)
			if err != nil {
				return nil, err
			}
			copy(outRow[c*e.weights.Cols:(c+1)*e.weights.Cols], e.weights.Data[idx*e.weights.Cols:(idx+1)*e.weights.Cols])
		}
	}
	return out, nil
}

func (e *Embedding) Backward(gradOutput *Matrix) (*Matrix, error) {
	if e.input == nil {
		return nil, fmt.Errorf("tensai: embedding backward called before forward")
	}
	wantCols := e.input.Cols * e.weights.Cols
	if gradOutput.Rows != e.input.Rows || gradOutput.Cols != wantCols {
		return nil, fmt.Errorf("tensai: embedding backward shape mismatch: grad %dx%d, want %dx%d",
			gradOutput.Rows, gradOutput.Cols, e.input.Rows, wantCols)
	}
	e.gradW = ensureMatrix(e.gradW, e.weights.Rows, e.weights.Cols)
	clear(e.gradW.Data)
	for r := 0; r < e.input.Rows; r++ {
		gRow := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		for c := 0; c < e.input.Cols; c++ {
			idx, err := embeddingIndex(e.input.At(r, c), e.weights.Rows)
			if err != nil {
				return nil, err
			}
			addSlice(e.gradW.Data[idx*e.weights.Cols:(idx+1)*e.weights.Cols], gRow[c*e.weights.Cols:(c+1)*e.weights.Cols])
		}
	}
	gradInput := e.bwdBuf(e.input.Rows, e.input.Cols)
	clear(gradInput.Data)
	return gradInput, nil
}

func (e *Embedding) Grads() (*Matrix, []Float) {
	return e.gradW, nil
}

func (e *Embedding) Params() (*Matrix, []Float) {
	return e.weights, nil
}

func (e *Embedding) SetParams(weights *Matrix, bias []Float) error {
	if weights == nil || len(bias) != 0 {
		return fmt.Errorf("tensai: embedding SetParams mismatch: weights=%v bias len=%d",
			shapeString(weights), len(bias))
	}
	e.weights = weights
	return nil
}

// Softmax normalizes each row into a probability distribution. Unlike the
// element-wise activations its backward pass couples all columns of a row.
// Note that SoftmaxCrossEntropy already applies softmax internally; use this
// layer only when the model output itself must be probabilities (e.g. with
// a custom loss).
type Softmax struct{ activationBase }

func (s *Softmax) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (s *Softmax) Forward(input *Matrix) (*Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	s.input = input
	out := s.fwdBuf(input.Rows, input.Cols)
	for r := 0; r < input.Rows; r++ {
		row := input.Data[r*input.Cols : (r+1)*input.Cols]
		outRow := out.Data[r*input.Cols : (r+1)*input.Cols]
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
		expShift(outRow, row, maxVal)
		var denom Float
		for _, e := range outRow {
			denom += e
		}
		scaleSlice(outRow, 1/denom)
	}
	s.output = out
	return out, nil
}

func (s *Softmax) Backward(gradOutput *Matrix) (*Matrix, error) {
	if s.output == nil {
		return nil, fmt.Errorf("tensai: softmax backward called before forward")
	}
	out := s.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	for r := 0; r < gradOutput.Rows; r++ {
		g := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		y := s.output.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		outRow := out.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		// dx_i = y_i * (g_i - sum_j g_j*y_j)
		var dot Float
		for i := range g {
			dot += g[i] * y[i]
		}
		for i := range g {
			outRow[i] = y[i] * (g[i] - dot)
		}
	}
	return out, nil
}

func geluF(x Float) Float {
	return 0.5 * x * (1 + Float(math.Erf(float64(x/math.Sqrt2))))
}

func geluGrad(x Float) Float {
	const invSqrt2Pi = 0.3989422804014327
	return 0.5*(1+Float(math.Erf(float64(x/math.Sqrt2)))) + x*Float(invSqrt2Pi)*expF(-0.5*x*x)
}

func shapeString(m *Matrix) string {
	if m == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%dx%d", m.Rows, m.Cols)
}

func embeddingIndex(v Float, vocabSize int) (int, error) {
	idx := int(v)
	if Float(idx) != v {
		return 0, fmt.Errorf("tensai: embedding token id must be an integer, got %g", v)
	}
	if idx < 0 || idx >= vocabSize {
		return 0, fmt.Errorf("tensai: embedding token id %d out of range [0,%d)", idx, vocabSize)
	}
	return idx, nil
}
