package llm

import (
	"math"
	"testing"
)

// fp4 is the FP4 (E2M1) codebook at its true magnitudes, against which
// mxfp4Table is doubled. Written out independently here so the test does
// not just restate the table it is checking.
var fp4 = [16]float64{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// packRow lays out one row of MXFP4 the way the checkpoint does: 16 bytes
// per group of 32 values, low nibble first, and one e8m0 exponent byte
// per group.
func packRow(codes []uint8, exps []uint8) (blocks, scales []byte) {
	groups := len(codes) / 32
	blocks = make([]byte, groups*16)
	for g := 0; g < groups; g++ {
		for i := 0; i < 16; i++ {
			lo := codes[g*32+2*i] & 0x0F
			hi := codes[g*32+2*i+1] & 0x0F
			blocks[g*16+i] = lo | hi<<4
		}
	}
	return blocks, append([]byte(nil), exps...)
}

func TestDequantMXFP4Row(t *testing.T) {
	// Two groups, every code used, two different exponents.
	codes := make([]uint8, 64)
	for i := range codes {
		codes[i] = uint8(i % 16)
	}
	exps := []uint8{127, 130} // 2^0 and 2^3 once the doubled table halves
	blocks, scales := packRow(codes, exps)
	got := make([]float32, 64)
	dequantMXFP4Row(blocks, scales, got)

	for i, c := range codes {
		scale := math.Ldexp(1, int(exps[i/32])-127)
		want := fp4[c] * scale
		if math.Abs(float64(got[i])-want) > 1e-6 {
			t.Fatalf("value %d (code %d, exp %d): got %g want %g", i, c, exps[i/32], got[i], want)
		}
	}
}

// TestMXFP4ExpertLayout pins both permutations that would otherwise fail
// silently: the [experts, out, groups, 16] indexing with its transpose to
// [in, out], and the de-interleaving of gpt-oss's fused gate|up.
func TestMXFP4ExpertLayout(t *testing.T) {
	const experts, out, groups = 3, 8, 2
	const in = groups * 32
	// Give every (expert, out-row, input) a distinguishable value: code
	// index walks so no two positions in a row share a magnitude
	// pattern, and each row gets its own exponent.
	blocks := make([]byte, experts*out*groups*16)
	scales := make([]byte, experts*out*groups)
	codeAt := func(e, r, k int) uint8 { return uint8((e*out+r+k)%15 + 1) } // never 0
	for e := 0; e < experts; e++ {
		for r := 0; r < out; r++ {
			codes := make([]uint8, in)
			for k := range codes {
				codes[k] = codeAt(e, r, k)
			}
			exps := make([]uint8, groups)
			for g := range exps {
				exps[g] = uint8(127 + (r+g)%3)
			}
			b, s := packRow(codes, exps)
			i := e*out + r
			copy(blocks[i*groups*16:], b)
			copy(scales[i*groups:], s)
		}
	}
	shape := []int{experts, out, groups, 16}

	want := func(e, r, k int) float64 {
		return fp4[codeAt(e, r, k)] * math.Ldexp(1, (r+k/32)%3)
	}

	// Plain (down_proj): out row r becomes column r.
	for _, e := range []int{0, 2} {
		w, err := mxfp4Expert(blocks, scales, shape, e, false)
		if err != nil {
			t.Fatal(err)
		}
		if w.Rows != in || w.Cols != out {
			t.Fatalf("shape = %dx%d, want %dx%d", w.Rows, w.Cols, in, out)
		}
		for r := 0; r < out; r++ {
			for k := 0; k < in; k++ {
				if got, exp := float64(w.Data[k*out+r]), want(e, r, k); math.Abs(got-exp) > 1e-6 {
					t.Fatalf("expert %d [in %d, out %d]: got %g want %g", e, k, r, got, exp)
				}
			}
		}
	}

	// Fused gate|up: row 2c is gate column c, row 2c+1 is up column
	// half+c. Getting this backwards swaps the SwiGLU's two operands,
	// which produces fluent nonsense rather than an error.
	const half = out / 2
	w, err := mxfp4Expert(blocks, scales, shape, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	for r := 0; r < out; r++ {
		col := r / 2
		if r%2 == 1 {
			col = half + r/2
		}
		for k := 0; k < in; k++ {
			if got, exp := float64(w.Data[k*out+col]), want(1, r, k); math.Abs(got-exp) > 1e-6 {
				t.Fatalf("fused row %d -> col %d [in %d]: got %g want %g", r, col, k, got, exp)
			}
		}
	}
}

func TestDeinterleaveBias(t *testing.T) {
	got := deinterleaveBias([]float32{0, 10, 1, 11, 2, 12})
	want := []float32{0, 1, 2, 10, 11, 12}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("deinterleaveBias = %v, want %v", got, want)
		}
	}
	if deinterleaveBias(nil) != nil {
		t.Error("deinterleaveBias(nil) should stay nil")
	}
}

func TestApplyGptOss(t *testing.T) {
	c := config{Intermediate: 2880}
	hf := gptOssConfig{NumLocalExperts: 32, ExpertsPerTok: 4}
	hf.RopeScaling = &struct {
		Type     string  `json:"rope_type"`
		Factor   float64 `json:"factor"`
		BetaFast float64 `json:"beta_fast"`
		BetaSlow float64 `json:"beta_slow"`
		OrigCtx  int     `json:"original_max_position_embeddings"`
	}{Type: "yarn", Factor: 32, BetaFast: 32, BetaSlow: 1, OrigCtx: 4096}
	applyGptOss(&c, hf)
	if c.NExpert != 32 || c.NExpertUsed != 4 || c.MoeFF != 2880 {
		t.Errorf("moe dims = %d/%d/%d, want 32/4/2880", c.NExpert, c.NExpertUsed, c.MoeFF)
	}
	if c.YarnFactor != 32 || c.YarnOrigCtx != 4096 || c.YarnBetaFast != 32 || c.YarnBetaSlow != 1 {
		t.Errorf("yarn = %v/%d/%v/%v", c.YarnFactor, c.YarnOrigCtx, c.YarnBetaFast, c.YarnBetaSlow)
	}
}

func TestGptOssSlides(t *testing.T) {
	hf := gptOssConfig{LayerTypes: []string{"sliding_attention", "full_attention", "sliding_attention"}}
	for i, want := range []bool{true, false, true} {
		if got := gptOssSlides(hf, i); got != want {
			t.Errorf("layer %d slides = %v, want %v", i, got, want)
		}
	}
	// Past the listed layers it falls back to the alternation.
	if !gptOssSlides(hf, 4) || gptOssSlides(hf, 5) {
		t.Error("fallback alternation is wrong")
	}
}
