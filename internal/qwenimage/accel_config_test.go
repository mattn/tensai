package qwenimage

import (
	"testing"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/gpu"
	"github.com/mattn/tensai/quant"
)

func TestProjectionBudget(t *testing.T) {
	l := &linear{q: quant.Quantize(tensai.NewMatrix(32, 32))}
	b := &Block{toQ: l, toK: l, toV: l, toOut: l, mlpProj: l, mlpGate: l, mlpOut: l}
	m := &Transformer{blocks: []*Block{b}}
	if err := UseGPUProjections(m, 1<<20); err == nil {
		t.Fatal("accepted a CPU-only model")
	}
	// Configuration validation must not allocate or touch the fake device.
	m.dev = &gpu.Device{}
	need := mlpBytes(b) + linearGPUBytes(l)
	if err := UseGPUProjections(m, need-1); err == nil || b.streamProjections {
		t.Fatal("exceeded budget")
	}
	if err := UseGPUProjections(m, need); err != nil || !b.streamProjections {
		t.Fatal("rejected exact budget", err)
	}
	b.streamProjections = false
	l.lora = &lowRank{a: tensai.NewMatrix(2, 32), b: tensai.NewMatrix(32, 2)}
	if err := UseGPUProjections(m, need); err == nil || b.streamProjections {
		t.Fatal("ignored streamed adapter weights")
	}
}

func TestGPUWeightBytes(t *testing.T) {
	w := tensai.NewMatrix(256, 32)
	q4, err := quant.Quantize4(w)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		l    *linear
		want uint64
	}{
		{&linear{q: quant.Quantize(w)}, 256*32 + 32*4},
		{&linear{q4: q4}, 128*32 + 4*32*4},
	} {
		if got := linearGPUBytes(tc.l); got != tc.want {
			t.Fatalf("%d != %d", got, tc.want)
		}
	}
}
