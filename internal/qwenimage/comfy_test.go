package qwenimage

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/mattn/tensai"
)

// hadamard builds ConvRot's matrix the way comfy-kitchen does: Kronecker
// powers of the 4x4 regular Hadamard, divided by sqrt(n).
func hadamard(n int) [][]float64 {
	h4 := [][]float64{{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}}
	h := h4
	for len(h) < n {
		m := len(h)
		k := make([][]float64, 4*m)
		for i := range k {
			k[i] = make([]float64, 4*m)
			for j := range k[i] {
				k[i][j] = h[i/4][j/4] * h4[i%4][j%4]
			}
		}
		h = k
	}
	for i := range h {
		for j := range h[i] {
			h[i][j] /= math.Sqrt(float64(n))
		}
	}
	return h
}

// The fast transform must be the product with the matrix itself.
func TestUnrotateIsHadamardProduct(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{4, 16, 64, 256} {
		h := hadamard(n)
		x := make([]tensai.Float, n)
		for i := range x {
			x[i] = tensai.Float(rng.NormFloat64())
		}
		want := make([]float64, n)
		for j := 0; j < n; j++ {
			for i := 0; i < n; i++ {
				want[j] += float64(x[i]) * h[i][j]
			}
		}
		got := append([]tensai.Float(nil), x...)
		unrotate(got)
		for j := range got {
			if math.Abs(float64(got[j])-want[j]) > 1e-5 {
				t.Fatalf("n=%d: element %d = %v, want %v", n, j, got[j], want[j])
			}
		}
	}
}

// rawTensor is one entry of a hand-built safetensors file.
type rawTensor struct {
	dtype string
	shape []int
	data  []byte
}

func writeSafetensors(t *testing.T, path string, ts map[string]rawTensor) {
	t.Helper()
	names := make([]string, 0, len(ts))
	for n := range ts {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]any{}
	var body []byte
	for _, n := range names {
		r := ts[n]
		header[n] = map[string]any{"dtype": r.dtype, "shape": r.shape, "data_offsets": []int{len(body), len(body) + len(r.data)}}
		body = append(body, r.data...)
	}
	h, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(append(out, h...), body...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

func f32Bytes(v []float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func bf16Bytes(v []float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint16(b, uint16(math.Float32bits(x)>>16))
	}
	return b
}

// quantizeConvRot stores w (rows x cols) the way comfy-kitchen does:
// rotate each group of columns by H^T, then int8 with a scale per row.
func quantizeConvRot(w []float64, rows, cols, group int) (q []byte, scale []float32) {
	h := hadamard(group)
	rot := make([]float64, len(w))
	for r := 0; r < rows; r++ {
		for g := 0; g < cols; g += group {
			for j := 0; j < group; j++ {
				var s float64
				for i := 0; i < group; i++ {
					s += w[r*cols+g+i] * h[j][i] // (W H^T)[j] = sum_i W[i] H[j][i]
				}
				rot[r*cols+g+j] = s
			}
		}
	}
	q = make([]byte, len(w))
	scale = make([]float32, rows)
	for r := 0; r < rows; r++ {
		var amax float64
		for _, x := range rot[r*cols : (r+1)*cols] {
			amax = math.Max(amax, math.Abs(x))
		}
		s := amax / 127
		scale[r] = float32(s)
		for c := 0; c < cols; c++ {
			q[r*cols+c] = byte(int8(math.Round(rot[r*cols+c] / s)))
		}
	}
	return q, scale
}

// A ComfyUI file reads back through diffusers names: int8 weights rotated
// back and rescaled, gate_up split gate first, the text encoder's
// language_model level restored, and embedding rows dequantized alone.
func TestComfyFile(t *testing.T) {
	const rows, cols, group = 6, 32, 16
	rng := rand.New(rand.NewPCG(3, 4))
	w := make([]float64, rows*cols)
	for i := range w {
		w[i] = rng.NormFloat64()
	}
	q, scale := quantizeConvRot(w, rows, cols, group)
	quant := []byte(`{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 16}`)
	gateUp := make([]float32, 4*cols) // two gate rows, then two proj rows
	for i := range gateUp {
		gateUp[i] = float32(i)
	}
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensors(t, path, map[string]rawTensor{
		"blk.weight":                            {"I8", []int{rows, cols}, q},
		"blk.weight_scale":                      {"F32", []int{rows, 1}, f32Bytes(scale)},
		"blk.comfy_quant":                       {"U8", []int{len(quant)}, quant},
		"b.0.img_mlp.gate_up.weight":            {"BF16", []int{4, cols}, bf16Bytes(gateUp)},
		"model.embed_tokens.weight":             {"I8", []int{rows, cols}, q},
		"model.embed_tokens.weight_scale":       {"F32", []int{rows, 1}, f32Bytes(scale)},
		"model.embed_tokens.comfy_quant":        {"U8", []int{len(quant)}, quant},
		"model.layers.0.input_layernorm.weight": {"F32", []int{2}, f32Bytes([]float32{1.5, 2.5})},
	})
	c, err := openComfy(path, comfyTextName)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	got, err := c.Tensor("blk.weight")
	if err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for i, x := range got.Data {
		num += (float64(x) - w[i]) * (float64(x) - w[i])
		den += w[i] * w[i]
	}
	if rel := math.Sqrt(num / den); rel > 0.02 {
		t.Fatalf("int8 weight comes back %.3g off (relative)", rel)
	}

	for name, first := range map[string]float32{
		"b.0.img_mlp.gate_layer.weight": 0,
		"b.0.img_mlp.proj.weight":       float32(2 * cols),
	} {
		half, err := c.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		if half.Shape[0] != 2 || half.Shape[1] != cols || half.Data[0] != first {
			t.Fatalf("%s: shape %v starting %v, want [2 %d] starting %v", name, half.Shape, half.Data[0], cols, first)
		}
	}

	norm, err := c.Tensor("model.language_model.layers.0.input_layernorm.weight")
	if err != nil || norm.Data[1] != 2.5 {
		t.Fatalf("renamed tensor: %v %v", norm, err)
	}

	emb, err := c.embedRows("model.language_model.embed_tokens.weight", []int{4, 1}, cols)
	if err != nil {
		t.Fatal(err)
	}
	for r, id := range []int{4, 1} {
		for j := 0; j < cols; j++ {
			if d := float64(emb.Data[r*cols+j]) - float64(got.Data[id*cols+j]); math.Abs(d) > 1e-6 {
				t.Fatalf("embedding row %d differs from the weight read whole", id)
			}
		}
	}
}

// Wan's VAE names map onto the diffusers ones the decoder reads; the
// full table was checked against both checkpoints, all 134 decoder
// tensors, and these are one of each kind.
func TestComfyVAEName(t *testing.T) {
	for diffusers, wan := range map[string]string{
		"post_quant_conv.weight":                             "conv2.weight",
		"decoder.conv_in.bias":                               "decoder.conv1.bias",
		"decoder.norm_out.gamma":                             "decoder.head.0.gamma",
		"decoder.conv_out.weight":                            "decoder.head.2.weight",
		"decoder.mid_block.attentions.0.to_qkv.weight":       "decoder.middle.1.to_qkv.weight",
		"decoder.mid_block.resnets.0.norm1.gamma":            "decoder.middle.0.residual.0.gamma",
		"decoder.mid_block.resnets.1.conv2.bias":             "decoder.middle.2.residual.6.bias",
		"decoder.up_blocks.2.resnets.0.conv_shortcut.weight": "decoder.upsamples.2.upsamples.0.shortcut.weight",
		"decoder.up_blocks.1.resnets.2.norm2.gamma":          "decoder.upsamples.1.upsamples.2.residual.3.gamma",
		"decoder.up_blocks.0.upsampler.resample.1.weight":    "decoder.upsamples.0.upsamples.3.resample.1.weight",
		"decoder.up_blocks.0.upsampler.time_conv.bias":       "decoder.upsamples.0.upsamples.3.time_conv.bias",
	} {
		if got := comfyVAEName(diffusers); got != wan {
			t.Errorf("%s -> %s, want %s", diffusers, got, wan)
		}
	}
}
