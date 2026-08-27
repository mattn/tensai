package layer

import (
	"fmt"
	"math/rand"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/kernels"
)

// Layer is a single differentiable stage of a Sequential model.
// Forward and Backward are batched: inputs/outputs are MxN matrices
// where M is the batch size and N is the feature dimension.
type Layer interface {
	// Init configures parameters using the given RNG and input width.
	Init(inputCols int, rng *rand.Rand) (outputCols int, err error)

	// Forward computes activations given the input batch.
	Forward(input *tensai.Matrix) (*tensai.Matrix, error)

	// Backward computes the gradient with respect to the layer input,
	// given the gradient with respect to the layer output.
	Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error)

	// Grads returns the parameter gradients accumulated during the last
	// backward pass, in the order [weights, bias]. Layers without
	// parameters return nil.
	Grads() (*tensai.Matrix, []tensai.Float)

	// Params returns the current parameters [weights, bias].
	Params() (*tensai.Matrix, []tensai.Float)

	// SetParams replaces the parameters [weights, bias].
	SetParams(weights *tensai.Matrix, bias []tensai.Float) error
}

// Dense is a fully-connected layer: y = x*W + b.
type Dense struct {
	bufferPair
	rng *rand.Rand

	weights *tensai.Matrix // (in x out)
	bias    []tensai.Float

	gradW *tensai.Matrix
	gradB []tensai.Float

	input *tensai.Matrix
	tW    *tensai.Matrix // scratch: weights^T
}

// NewDense returns a Dense layer with the given output size.
func NewDense(outCols int) *Dense {
	return &Dense{bias: make([]tensai.Float, outCols)}
}

func (d *Dense) Init(inputCols int, rng *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: dense init with non-positive input cols: %d", inputCols)
	}
	d.rng = rng
	d.weights = tensai.RandomMatrix(inputCols, d.outCols(), rng)
	d.bias = make([]tensai.Float, d.outCols())
	d.gradW = tensai.NewMatrix(inputCols, d.outCols())
	d.gradB = make([]tensai.Float, d.outCols())
	return d.outCols(), nil
}

func (d *Dense) outCols() int {
	if d.weights != nil {
		return d.weights.Cols
	}
	// Before Init, bias length is the configured output width.
	return len(d.bias)
}

func (d *Dense) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if input.Cols != d.weights.Rows {
		return nil, fmt.Errorf("tensai: dense forward shape mismatch: input %dx%d, weights %dx%d",
			input.Rows, input.Cols, d.weights.Rows, d.weights.Cols)
	}
	d.input = input
	out := d.fwdBuf(input.Rows, d.weights.Cols)
	if err := tensai.DotInto(out, input, d.weights); err != nil {
		return nil, err
	}
	for r := 0; r < out.Rows; r++ {
		kernels.AddSlice(out.Data[r*out.Cols:(r+1)*out.Cols], d.bias)
	}
	return out, nil
}

func (d *Dense) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if d.input == nil {
		return nil, fmt.Errorf("tensai: dense backward called before forward")
	}
	// gradW = input^T * gradOutput  (in x out), without materializing
	// the transpose.
	d.gradW = tensai.EnsureMatrix(d.gradW, d.weights.Rows, d.weights.Cols)
	if err := tensai.DotTAInto(d.gradW, d.input, gradOutput); err != nil {
		return nil, err
	}

	// gradB = sum over batch of gradOutput
	clear(d.gradB)
	for r := 0; r < gradOutput.Rows; r++ {
		kernels.AddSlice(d.gradB, gradOutput.Data[r*gradOutput.Cols:(r+1)*gradOutput.Cols])
	}

	// gradInput = gradOutput * weights^T  (batch x in)
	d.tW = tensai.EnsureMatrix(d.tW, d.weights.Cols, d.weights.Rows)
	if err := tensai.TInto(d.tW, d.weights); err != nil {
		return nil, err
	}
	gradInput := d.bwdBuf(gradOutput.Rows, d.weights.Rows)
	if err := tensai.DotInto(gradInput, gradOutput, d.tW); err != nil {
		return nil, err
	}
	return gradInput, nil
}

func (d *Dense) Grads() (*tensai.Matrix, []tensai.Float) {
	return d.gradW, d.gradB
}

func (d *Dense) Params() (*tensai.Matrix, []tensai.Float) {
	return d.weights, d.bias
}

func (d *Dense) SetParams(weights *tensai.Matrix, bias []tensai.Float) error {
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
	apply(x tensai.Float) tensai.Float
	applyGrad(x tensai.Float) tensai.Float
}

// bufferPair provides scratch matrices reused across training steps.
// Forward buffers are only reused while training (FitStep switches the mode
// on): Predict results stay valid until the next Predict of the same shape
// would otherwise silently overwrite them. Backward runs only during
// training, so its buffer is always reused.
type bufferPair struct {
	training bool
	fwd, bwd *tensai.Matrix
}

func (b *bufferPair) SetTraining(v bool) { b.training = v }

func (b *bufferPair) fwdBuf(rows, cols int) *tensai.Matrix {
	if !b.training {
		return tensai.NewMatrix(rows, cols)
	}
	b.fwd = tensai.EnsureMatrix(b.fwd, rows, cols)
	return b.fwd
}

func (b *bufferPair) bwdBuf(rows, cols int) *tensai.Matrix {
	b.bwd = tensai.EnsureMatrix(b.bwd, rows, cols)
	return b.bwd
}

// activationBase stores the last forward input/output for reuse in backward.
type activationBase struct {
	bufferPair
	input  *tensai.Matrix
	output *tensai.Matrix
}

func (a *activationBase) Params() (*tensai.Matrix, []tensai.Float)       { return nil, nil }
func (a *activationBase) SetParams(*tensai.Matrix, []tensai.Float) error { return nil }
func (a *activationBase) Grads() (*tensai.Matrix, []tensai.Float)        { return nil, nil }

// ReLU activation: f(x) = max(0, x).
type ReLU struct{ activationBase }

func (r *ReLU) apply(x tensai.Float) tensai.Float {
	if x > 0 {
		return x
	}
	return 0
}
func (r *ReLU) applyGrad(x tensai.Float) tensai.Float {
	if x > 0 {
		return 1
	}
	return 0
}

func (r *ReLU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (r *ReLU) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	r.input = input
	out := r.fwdBuf(input.Rows, input.Cols)
	kernels.ReluFwd(out.Data, input.Data)
	r.output = out
	return out, nil
}

func (r *ReLU) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if r.input == nil {
		return nil, fmt.Errorf("tensai: relu backward called before forward")
	}
	out := r.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	kernels.ReluBwd(out.Data, gradOutput.Data, r.input.Data)
	return out, nil
}

// LeakyReLU activation: f(x) = x for x > 0, alpha*x otherwise.
type LeakyReLU struct {
	activationBase
	Alpha tensai.Float
}

// NewLeakyReLU returns a LeakyReLU with the given negative-side slope.
func NewLeakyReLU(alpha tensai.Float) *LeakyReLU {
	return &LeakyReLU{Alpha: alpha}
}

func (l *LeakyReLU) apply(x tensai.Float) tensai.Float {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}
func (l *LeakyReLU) applyGrad(x tensai.Float) tensai.Float {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

func (l *LeakyReLU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (l *LeakyReLU) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	l.input = input
	out := l.fwdBuf(input.Rows, input.Cols)
	kernels.LeakyFwd(out.Data, input.Data, l.Alpha)
	l.output = out
	return out, nil
}

func (l *LeakyReLU) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if l.input == nil {
		return nil, fmt.Errorf("tensai: leakyrelu backward called before forward")
	}
	out := l.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	kernels.LeakyBwd(out.Data, gradOutput.Data, l.input.Data, l.Alpha)
	return out, nil
}

// Sigmoid activation: f(x) = 1 / (1 + e^-x).
type Sigmoid struct{ activationBase }

func (s *Sigmoid) apply(x tensai.Float) tensai.Float {
	return 1.0 / (1.0 + kernels.ExpF(-x))
}
func (s *Sigmoid) applyGrad(x tensai.Float) tensai.Float {
	// derivative in terms of the output y = sigmoid(x): y*(1-y)
	y := s.apply(x)
	return y * (1 - y)
}

func (s *Sigmoid) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (s *Sigmoid) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	s.input = input
	out := s.fwdBuf(input.Rows, input.Cols)
	kernels.SigmoidFwd(out.Data, input.Data)
	s.output = out
	return out, nil
}

func (s *Sigmoid) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if s.output == nil {
		return nil, fmt.Errorf("tensai: sigmoid backward called before forward")
	}
	out := s.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	kernels.SigmoidBwd(out.Data, gradOutput.Data, s.output.Data)
	return out, nil
}

// Tanh activation: f(x) = tanh(x).
type Tanh struct{ activationBase }

func (t *Tanh) apply(x tensai.Float) tensai.Float { return kernels.TanhF(x) }
func (t *Tanh) applyGrad(x tensai.Float) tensai.Float {
	y := t.apply(x)
	return 1 - y*y
}

func (t *Tanh) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (t *Tanh) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	t.input = input
	out := t.fwdBuf(input.Rows, input.Cols)
	kernels.TanhFwd(out.Data, input.Data)
	t.output = out
	return out, nil
}

func (t *Tanh) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if t.output == nil {
		return nil, fmt.Errorf("tensai: tanh backward called before forward")
	}
	out := t.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	kernels.TanhBwd(out.Data, gradOutput.Data, t.output.Data)
	return out, nil
}

// GELU activation: f(x) = 0.5*x*(1+erf(x/sqrt(2))).
type GELU struct{ activationBase }

func (g *GELU) Init(inputCols int, _ *rand.Rand) (int, error) { return inputCols, nil }

func (g *GELU) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	g.input = input
	out := g.fwdBuf(input.Rows, input.Cols)
	kernels.GeluFwd(out.Data, input.Data)
	g.output = out
	return out, nil
}

func (g *GELU) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if g.input == nil {
		return nil, fmt.Errorf("tensai: gelu backward called before forward")
	}
	out := g.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	kernels.GeluBwd(out.Data, gradOutput.Data, g.input.Data)
	return out, nil
}

// LayerNorm normalizes each row over its feature dimension and applies a
// learnable affine transform.
type LayerNorm struct {
	bufferPair

	gamma *tensai.Matrix
	beta  []tensai.Float

	gradGamma *tensai.Matrix
	gradBeta  []tensai.Float

	normalized *tensai.Matrix
	invStd     []tensai.Float
	eps        tensai.Float
}

// NewLayerNorm returns a LayerNorm with the default epsilon.
func NewLayerNorm() *LayerNorm {
	return &LayerNorm{eps: 1e-5}
}

func (l *LayerNorm) Init(inputCols int, _ *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: layernorm init with non-positive input cols: %d", inputCols)
	}
	l.gamma = tensai.NewMatrix(1, inputCols)
	for i := range l.gamma.Data {
		l.gamma.Data[i] = 1
	}
	l.beta = make([]tensai.Float, inputCols)
	l.gradGamma = tensai.NewMatrix(1, inputCols)
	l.gradBeta = make([]tensai.Float, inputCols)
	return inputCols, nil
}

func (l *LayerNorm) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
	if err := input.Validate(); err != nil {
		return nil, err
	}
	if l.gamma == nil || input.Cols != l.gamma.Cols {
		return nil, fmt.Errorf("tensai: layernorm forward shape mismatch: input %dx%d gamma %v",
			input.Rows, input.Cols, shapeString(l.gamma))
	}
	out := l.fwdBuf(input.Rows, input.Cols)
	l.normalized = tensai.EnsureMatrix(l.normalized, input.Rows, input.Cols)
	if cap(l.invStd) < input.Rows {
		l.invStd = make([]tensai.Float, input.Rows)
	} else {
		l.invStd = l.invStd[:input.Rows]
	}
	for r := 0; r < input.Rows; r++ {
		inRow := input.Data[r*input.Cols : (r+1)*input.Cols]
		normRow := l.normalized.Data[r*input.Cols : (r+1)*input.Cols]
		outRow := out.Data[r*input.Cols : (r+1)*input.Cols]
		l.invStd[r] = kernels.LnFwdRow(outRow, normRow, inRow, l.gamma.Data, l.beta, l.eps)
	}
	return out, nil
}

func (l *LayerNorm) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if l.normalized == nil {
		return nil, fmt.Errorf("tensai: layernorm backward called before forward")
	}
	l.gradGamma = tensai.EnsureMatrix(l.gradGamma, 1, gradOutput.Cols)
	clear(l.gradGamma.Data)
	clear(l.gradBeta)
	out := l.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	for r := 0; r < gradOutput.Rows; r++ {
		gRow := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		xhatRow := l.normalized.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		outRow := out.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		kernels.LnBwdRow(outRow, gRow, xhatRow, l.gamma.Data, l.gradGamma.Data, l.gradBeta, l.invStd[r])
	}
	return out, nil
}

func (l *LayerNorm) Grads() (*tensai.Matrix, []tensai.Float) {
	return l.gradGamma, l.gradBeta
}

func (l *LayerNorm) Params() (*tensai.Matrix, []tensai.Float) {
	return l.gamma, l.beta
}

func (l *LayerNorm) SetParams(weights *tensai.Matrix, bias []tensai.Float) error {
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

	weights *tensai.Matrix
	gradW   *tensai.Matrix
	input   *tensai.Matrix
}

// NewEmbedding returns a trainable embedding table of shape vocabSize x dim.
func NewEmbedding(vocabSize, dim int) *Embedding {
	return &Embedding{weights: tensai.NewMatrix(vocabSize, dim)}
}

func (e *Embedding) Init(inputCols int, rng *rand.Rand) (int, error) {
	if inputCols <= 0 {
		return 0, fmt.Errorf("tensai: embedding init with non-positive input cols: %d", inputCols)
	}
	if e.weights == nil || e.weights.Rows <= 0 || e.weights.Cols <= 0 {
		return 0, fmt.Errorf("tensai: embedding init with invalid table shape")
	}
	e.weights = tensai.RandomMatrix(e.weights.Rows, e.weights.Cols, rng)
	e.gradW = tensai.NewMatrix(e.weights.Rows, e.weights.Cols)
	return inputCols * e.weights.Cols, nil
}

func (e *Embedding) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
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

func (e *Embedding) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if e.input == nil {
		return nil, fmt.Errorf("tensai: embedding backward called before forward")
	}
	wantCols := e.input.Cols * e.weights.Cols
	if gradOutput.Rows != e.input.Rows || gradOutput.Cols != wantCols {
		return nil, fmt.Errorf("tensai: embedding backward shape mismatch: grad %dx%d, want %dx%d",
			gradOutput.Rows, gradOutput.Cols, e.input.Rows, wantCols)
	}
	e.gradW = tensai.EnsureMatrix(e.gradW, e.weights.Rows, e.weights.Cols)
	clear(e.gradW.Data)
	for r := 0; r < e.input.Rows; r++ {
		gRow := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		for c := 0; c < e.input.Cols; c++ {
			idx, err := embeddingIndex(e.input.At(r, c), e.weights.Rows)
			if err != nil {
				return nil, err
			}
			kernels.AddSlice(e.gradW.Data[idx*e.weights.Cols:(idx+1)*e.weights.Cols], gRow[c*e.weights.Cols:(c+1)*e.weights.Cols])
		}
	}
	gradInput := e.bwdBuf(e.input.Rows, e.input.Cols)
	clear(gradInput.Data)
	return gradInput, nil
}

func (e *Embedding) Grads() (*tensai.Matrix, []tensai.Float) {
	return e.gradW, nil
}

func (e *Embedding) Params() (*tensai.Matrix, []tensai.Float) {
	return e.weights, nil
}

func (e *Embedding) SetParams(weights *tensai.Matrix, bias []tensai.Float) error {
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

func (s *Softmax) Forward(input *tensai.Matrix) (*tensai.Matrix, error) {
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
		kernels.ExpShift(outRow, row, maxVal)
		var denom tensai.Float
		for _, e := range outRow {
			denom += e
		}
		kernels.ScaleSlice(outRow, 1/denom)
	}
	s.output = out
	return out, nil
}

func (s *Softmax) Backward(gradOutput *tensai.Matrix) (*tensai.Matrix, error) {
	if s.output == nil {
		return nil, fmt.Errorf("tensai: softmax backward called before forward")
	}
	out := s.bwdBuf(gradOutput.Rows, gradOutput.Cols)
	for r := 0; r < gradOutput.Rows; r++ {
		g := gradOutput.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		y := s.output.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		outRow := out.Data[r*gradOutput.Cols : (r+1)*gradOutput.Cols]
		// dx_i = y_i * (g_i - sum_j g_j*y_j)
		var dot tensai.Float
		for i := range g {
			dot += g[i] * y[i]
		}
		for i := range g {
			outRow[i] = y[i] * (g[i] - dot)
		}
	}
	return out, nil
}

func shapeString(m *tensai.Matrix) string {
	if m == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%dx%d", m.Rows, m.Cols)
}

func embeddingIndex(v tensai.Float, vocabSize int) (int, error) {
	idx := int(v)
	if tensai.Float(idx) != v {
		return 0, fmt.Errorf("tensai: embedding token id must be an integer, got %g", v)
	}
	if idx < 0 || idx >= vocabSize {
		return 0, fmt.Errorf("tensai: embedding token id %d out of range [0,%d)", idx, vocabSize)
	}
	return idx, nil
}
