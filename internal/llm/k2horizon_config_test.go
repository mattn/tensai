package llm

import "testing"

// K2-Horizon-0.9B: a dense decoder-only model with IFM's pipe-delimited
// chat markers (<|ifm|im_start|>, <ifm|think>).  Unlike qwen3_5 it has
// no interleaved linear-attention layers or output gate.
func TestLoadK2HorizonConfig(t *testing.T) {
	c, err := loadConfig("testdata/k2-horizon-config.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		got  any
		want any
	}{
		{"model_type", c.ModelType, "k2-horizon"},
		{"layers", c.Layers, 28},
		{"hidden", c.HiddenSize, 1536},
		{"head_dim", c.HeadDim, 64},
		{"heads", c.Heads, 32},
		{"kv heads", c.KVHeads, 8},
		{"intermediate", c.Intermediate, 5120},
		{"vocab", c.Vocab, 151936},
		{"tied", c.TieEmbedding, true},
		{"rope theta", c.RopeTheta, 1000000.0},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %v, want %v", tt.name, tt.got, tt.want)
		}
	}
}
