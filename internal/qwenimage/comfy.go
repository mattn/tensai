package qwenimage

// ComfyUI's repackaging of Qwen-Image-2.1 (Comfy-Org/Qwen-Image-2.1) holds
// the same three models as the diffusers checkpoint, one file apiece,
// under names and in forms of its own:
//
//   - the transformer fuses each block's img_mlp.gate_layer and
//     img_mlp.proj into one img_mlp.gate_up, gate rows first;
//   - the text encoder drops "language_model." from its layer names;
//   - the VAE keeps Wan's original names (decoder.middle.0.residual.2,
//     not decoder.mid_block.resnets.0.conv1) and its convolutions keep a
//     time axis of one;
//   - the int8 files store each weight as int8 with a scale per row, after
//     rotating every 256 input columns by a Hadamard matrix ("convrot") so
//     outliers spread out before rounding.
//
// comfyFile reads such a file through the diffusers names, so every loader
// here works on it unchanged: a weight comes back as float32, rotated back
// and rescaled, and the loaders quantize it their own way as usual.

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"runtime"
	"strings"
	"sync"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/encoding/safetensors"
)

// comfyFile is one ComfyUI weights file seen through diffusers names.
type comfyFile struct {
	f *safetensors.File
	// rename maps a diffusers name to the file's own.
	rename func(string) string
	// fused holds the last gate_up read: its two halves are asked for one
	// after the other, and it is half a gigabyte as floats.
	fusedName string
	fused     *tensai.Tensor
	fusedUsed int
}

func openComfy(path string, rename func(string) string) (*comfyFile, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	if rename == nil {
		rename = func(s string) string { return s }
	}
	return &comfyFile{f: f, rename: rename}, nil
}

func (c *comfyFile) Close() error { return c.f.Close() }

// Tensor returns the named diffusers weight as float32.
func (c *comfyFile) Tensor(name string) (*tensai.Tensor, error) {
	for _, half := range []struct {
		suffix string
		index  int
	}{{"img_mlp.gate_layer.weight", 0}, {"img_mlp.proj.weight", 1}} {
		if !strings.HasSuffix(name, half.suffix) {
			continue
		}
		fused := strings.TrimSuffix(name, half.suffix) + "img_mlp.gate_up.weight"
		if _, _, ok := c.f.Info(fused); !ok {
			break
		}
		return c.fusedHalf(fused, half.index)
	}
	return c.read(c.rename(name))
}

// fusedHalf returns rows [index*n/2, (index+1)*n/2) of a fused weight.
func (c *comfyFile) fusedHalf(name string, index int) (*tensai.Tensor, error) {
	if c.fusedName != name {
		t, err := c.read(name)
		if err != nil {
			return nil, err
		}
		c.fusedName, c.fused, c.fusedUsed = name, t, 0
	}
	t := c.fused
	if len(t.Shape) != 2 || t.Shape[0]%2 != 0 {
		return nil, fmt.Errorf("qwenimage: %s has shape %v, not two stacked halves", name, t.Shape)
	}
	rows, cols := t.Shape[0]/2, t.Shape[1]
	out := tensai.NewTensor(rows, cols)
	copy(out.Data, t.Data[index*rows*cols:(index+1)*rows*cols])
	if c.fusedUsed++; c.fusedUsed == 2 {
		c.fusedName, c.fused = "", nil
	}
	return out, nil
}

// comfyQuant is a quantized weight's comfy_quant entry.
type comfyQuant struct {
	Format    string `json:"format"`
	ConvRot   bool   `json:"convrot"`
	GroupSize int    `json:"convrot_groupsize"`
}

// read returns a tensor by the file's own name, dequantizing an int8 one.
func (c *comfyFile) read(name string) (*tensai.Tensor, error) {
	dtype, shape, ok := c.f.Info(name)
	if !ok {
		return nil, fmt.Errorf("qwenimage: %s is not in the checkpoint", name)
	}
	if dtype != "I8" {
		return c.f.Tensor(name)
	}
	base := strings.TrimSuffix(name, ".weight")
	q, err := c.quant(base)
	if err != nil {
		return nil, err
	}
	raw, _, err := c.f.Raw(name)
	if err != nil {
		return nil, err
	}
	scale, err := c.f.Tensor(base + ".weight_scale")
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("qwenimage: int8 %s has shape %v", name, shape)
	}
	rows, cols := shape[0], shape[1]
	if len(scale.Data) != rows && len(scale.Data) != 1 {
		return nil, fmt.Errorf("qwenimage: %s has %d scales for %d rows", name, len(scale.Data), rows)
	}
	out := tensai.NewTensor(rows, cols)
	parallelRows(rows, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			dequantRow(out.Data[r*cols:(r+1)*cols], raw[r*cols:(r+1)*cols], scaleOf(scale.Data, r), q)
		}
	})
	return out, nil
}

// quant reads a weight's comfy_quant entry and checks it is a form this
// reader knows.
func (c *comfyFile) quant(base string) (comfyQuant, error) {
	raw, _, err := c.f.Raw(base + ".comfy_quant")
	if err != nil {
		return comfyQuant{}, fmt.Errorf("qwenimage: int8 %s has no comfy_quant entry", base)
	}
	var q comfyQuant
	if err := json.Unmarshal(raw, &q); err != nil {
		return q, fmt.Errorf("qwenimage: %s.comfy_quant: %w", base, err)
	}
	if q.Format != "int8_tensorwise" {
		return q, fmt.Errorf("qwenimage: %s is quantized as %q, which this reader does not know", base, q.Format)
	}
	if q.ConvRot && !isPowerOf4(q.GroupSize) {
		return q, fmt.Errorf("qwenimage: %s rotates in groups of %d, not a power of four", base, q.GroupSize)
	}
	return q, nil
}

func scaleOf(s []tensai.Float, r int) tensai.Float {
	if len(s) == 1 {
		return s[0]
	}
	return s[r]
}

// dequantRow writes one row of an int8 weight back as floats: the stored
// values times the row's scale, then, for a rotated weight, each group of
// columns rotated back.
func dequantRow(dst []tensai.Float, q []byte, scale tensai.Float, cq comfyQuant) {
	for i, b := range q {
		dst[i] = tensai.Float(int8(b)) * scale
	}
	if !cq.ConvRot {
		return
	}
	for g := 0; g+cq.GroupSize <= len(dst); g += cq.GroupSize {
		hadamard(dst[g : g+cq.GroupSize])
	}
}

// hadamard multiplies a group by ConvRot's Hadamard matrix H, the
// Kronecker power of the 4x4 regular Hadamard below scaled to be
// orthonormal. H is symmetric and its own inverse, so the one product
// both rotates and rotates back: a ComfyUI weight was stored as W H^T,
// and this undoes it. H is applied one base-4 digit of the index at a
// time, 4 adds a value per digit rather than a group-wide product.
func hadamard(x []tensai.Float) {
	n := len(x)
	for s := 1; s < n; s *= 4 {
		for b := 0; b < n; b += 4 * s {
			for o := b; o < b+s; o++ {
				x0, x1, x2, x3 := x[o], x[o+s], x[o+2*s], x[o+3*s]
				x[o] = x0 + x1 + x2 - x3
				x[o+s] = x0 + x1 - x2 + x3
				x[o+2*s] = x0 - x1 + x2 + x3
				x[o+3*s] = -x0 + x1 + x2 + x3
			}
		}
	}
	norm := tensai.Float(1 / math.Sqrt(float64(n)))
	for i := range x {
		x[i] *= norm
	}
}

func isPowerOf4(n int) bool {
	for n > 1 && n%4 == 0 {
		n /= 4
	}
	return n == 1
}

// parallelRows splits rows across the CPUs.
func parallelRows(rows int, fn func(lo, hi int)) {
	workers := min(runtime.NumCPU(), rows)
	if workers <= 1 {
		fn(0, rows)
		return
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < rows; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, min(lo+chunk, rows))
	}
	wg.Wait()
}

// embedRows reads the rows ids of an embedding table by its file name,
// dequantizing just those rows when the table is int8.
func (c *comfyFile) embedRows(name string, ids []int, dim int) (*tensai.Matrix, error) {
	name = c.rename(name)
	dtype, shape, ok := c.f.Info(name)
	if !ok {
		return nil, fmt.Errorf("qwenimage: %s is not in the checkpoint", name)
	}
	if len(shape) != 2 || shape[1] != dim {
		return nil, fmt.Errorf("qwenimage: embedding table has shape %v", shape)
	}
	raw, _, err := c.f.Raw(name)
	if err != nil {
		return nil, err
	}
	out := tensai.NewMatrix(len(ids), dim)
	switch dtype {
	case "BF16":
		return embedBF16(raw, shape[0], dim, ids)
	case "I8":
		base := strings.TrimSuffix(name, ".weight")
		q, err := c.quant(base)
		if err != nil {
			return nil, err
		}
		scale, err := c.f.Tensor(base + ".weight_scale")
		if err != nil {
			return nil, err
		}
		for r, id := range ids {
			if id < 0 || id >= shape[0] {
				return nil, fmt.Errorf("qwenimage: token %d is outside the %d-row table", id, shape[0])
			}
			dequantRow(out.Data[r*dim:(r+1)*dim], raw[id*dim:(id+1)*dim], scaleOf(scale.Data, id), q)
		}
		return out, nil
	}
	return nil, fmt.Errorf("qwenimage: embedding table is %s", dtype)
}

// comfyTextName maps the diffusers text encoder names onto ComfyUI's,
// which drop the language_model level.
func comfyTextName(name string) string {
	return strings.Replace(name, "model.language_model.", "model.", 1)
}

var (
	comfyVAEMid   = regexp.MustCompile(`^decoder\.mid_block\.resnets\.(\d+)\.(.*)$`)
	comfyVAEUp    = regexp.MustCompile(`^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)$`)
	comfyVAEUpper = regexp.MustCompile(`^decoder\.up_blocks\.(\d+)\.upsampler\.(.*)$`)
)

// comfyVAEResidual maps a diffusers resnet part onto Wan's residual
// sequence, whose indexes count the norm, activation and dropout
// layers between the convolutions.
var comfyVAEResidual = map[string]string{
	"norm1.gamma":          "residual.0.gamma",
	"conv1.weight":         "residual.2.weight",
	"conv1.bias":           "residual.2.bias",
	"norm2.gamma":          "residual.3.gamma",
	"conv2.weight":         "residual.6.weight",
	"conv2.bias":           "residual.6.bias",
	"conv_shortcut.weight": "shortcut.weight",
	"conv_shortcut.bias":   "shortcut.bias",
}

// comfyVAEName maps a diffusers VAE decoder name onto Wan's.
func comfyVAEName(name string) string {
	switch {
	case strings.HasPrefix(name, "post_quant_conv."):
		return "conv2." + strings.TrimPrefix(name, "post_quant_conv.")
	case strings.HasPrefix(name, "decoder.conv_in."):
		return "decoder.conv1." + strings.TrimPrefix(name, "decoder.conv_in.")
	case name == "decoder.norm_out.gamma":
		return "decoder.head.0.gamma"
	case strings.HasPrefix(name, "decoder.conv_out."):
		return "decoder.head.2." + strings.TrimPrefix(name, "decoder.conv_out.")
	case strings.HasPrefix(name, "decoder.mid_block.attentions.0."):
		return "decoder.middle.1." + strings.TrimPrefix(name, "decoder.mid_block.attentions.0.")
	}
	if m := comfyVAEMid.FindStringSubmatch(name); m != nil {
		// middle.0 and middle.2 are the resnets either side of the
		// attention at middle.1.
		i := 0
		if m[1] == "1" {
			i = 2
		}
		return fmt.Sprintf("decoder.middle.%d.%s", i, comfyVAEResidual[m[2]])
	}
	if m := comfyVAEUp.FindStringSubmatch(name); m != nil {
		return fmt.Sprintf("decoder.upsamples.%s.upsamples.%s.%s", m[1], m[2], comfyVAEResidual[m[3]])
	}
	if m := comfyVAEUpper.FindStringSubmatch(name); m != nil {
		// The upsampler follows each stage's three resnets.
		return fmt.Sprintf("decoder.upsamples.%s.upsamples.3.%s", m[1], m[2])
	}
	return name
}

// comfyVAE reads a ComfyUI VAE through diffusers names, dropping the
// time axis of one its convolutions keep.
type comfyVAE struct{ f *safetensors.File }

func (v comfyVAE) Tensor(name string) (*tensai.Tensor, error) {
	t, err := v.f.Tensor(comfyVAEName(name))
	if err != nil {
		return nil, err
	}
	if len(t.Shape) == 5 && t.Shape[2] == 1 {
		t.Shape = []int{t.Shape[0], t.Shape[1], t.Shape[3], t.Shape[4]}
	}
	return t, nil
}

func (v comfyVAE) Info(name string) (string, []int, bool) {
	return v.f.Info(comfyVAEName(name))
}

// isComfyVAE reports whether a VAE file uses Wan's names.
func isComfyVAE(f *safetensors.File) bool {
	_, _, ok := f.Info("decoder.conv1.weight")
	return ok
}
