package llm

import "testing"

// The real Qwen3.5-0.8B config: everything the text model needs sits one
// level down, beside a vision tower the loader has no use for.
func TestLoadQwen35Config(t *testing.T) {
	c, err := loadConfig("testdata/qwen3_5-config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		got  any
		want any
	}{
		{"model_type", c.ModelType, "qwen3_5"},
		{"prefix", c.Prefix, "language_model."},
		{"layers", c.Layers, 24},
		{"hidden", c.HiddenSize, 1024},
		{"head_dim", c.HeadDim, 256},
		{"heads", c.Heads, 8},
		{"kv heads", c.KVHeads, 2},
		{"intermediate", c.Intermediate, 3584},
		{"vocab", c.Vocab, 248320},
		{"tied", c.TieEmbedding, true},
		{"rope theta", c.RopeTheta, 10000000.0},
		{"partial rotary", c.PartialRotary, 0.25},
		{"output gate", c.AttnOutputGate, true},
		{"full attention interval", c.FullAttnInterval, 4},
		{"linear key heads", c.LinearKeyHeads, 16},
		{"linear value heads", c.LinearValueHeads, 16},
		{"linear key dim", c.LinearKeyDim, 128},
		{"linear value dim", c.LinearValueDim, 128},
		{"conv kernel", c.LinearConvK, 4},
		{"layer types", len(c.LayerTypes), 24},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %v, want %v", tt.name, tt.got, tt.want)
		}
	}
	// Every fourth layer is the one with a KV cache.
	full := 0
	for i, lt := range c.LayerTypes {
		if lt == "full_attention" {
			full++
			if (i+1)%c.FullAttnInterval != 0 {
				t.Errorf("layer %d is full attention but not on the interval", i)
			}
		}
	}
	if full != 6 {
		t.Errorf("got %d full-attention layers, want 6", full)
	}
}
