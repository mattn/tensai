package llm

import "testing"

func TestCommonPrefix(t *testing.T) {
	for _, tt := range []struct {
		a, b []int
		want int
	}{
		{[]int{1, 2, 3}, []int{1, 2, 3}, 3},
		{[]int{1, 2, 3}, []int{1, 2, 9}, 2},
		{[]int{1, 2}, []int{1, 2, 3}, 2},
		{[]int{9}, []int{1}, 0},
		{nil, []int{1}, 0},
	} {
		if got := commonPrefix(tt.a, tt.b); got != tt.want {
			t.Errorf("commonPrefix(%v,%v) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestPlan(t *testing.T) {
	sys := []int{1, 2, 3, 4} // the shared opening an agent resends
	for _, tt := range []struct {
		name        string
		c           promptCache
		tokens      []int
		wantStart   int
		wantRestore bool
	}{
		{"disabled", promptCache{live: sys}, append(sys, 5), 0, false},
		{"nothing cached", promptCache{enabled: true}, sys, 0, false},
		// Continuing where the last turn stopped needs no rollback, which
		// is the one case a recurrent state can serve too.
		{"extends the live cache", promptCache{enabled: true, live: sys, hasDelta: true},
			append(append([]int{}, sys...), 5, 6), 4, false},
		// A KV cache rolls back; a recurrent state does not.
		{"diverges, attention only", promptCache{enabled: true, live: append(append([]int{}, sys...), 9)},
			append(append([]int{}, sys...), 5), 4, false},
		{"diverges, recurrent", promptCache{enabled: true, hasDelta: true,
			live: append(append([]int{}, sys...), 9)}, append(append([]int{}, sys...), 5), 0, false},
		// ... unless a checkpoint was taken where they parted.
		{"diverges, recurrent, checkpointed", promptCache{enabled: true, hasDelta: true,
			live: append(append([]int{}, sys...), 9), ckpt: sys},
			append(append([]int{}, sys...), 5), 4, true},
		// An identical prompt has nothing left to prefill and must not
		// claim the whole thing is already done.
		{"identical", promptCache{enabled: true, live: sys}, sys, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			start, restore := tt.c.plan(tt.tokens)
			if start != tt.wantStart || restore != tt.wantRestore {
				t.Errorf("plan = (%d, %v), want (%d, %v)", start, restore, tt.wantStart, tt.wantRestore)
			}
		})
	}
}

// A snapshot must survive the state moving on, and restore it exactly.
func TestDeltaSnapshotRoundTrip(t *testing.T) {
	const hidden, heads, kd, vd, convK = 8, 2, 4, 4, 4
	r := &lcg{x: 7}
	d := &deltaWeights{heads: heads, kDim: kd, vDim: vd, convK: convK}
	d.convDim = kd*heads*2 + vd*heads
	d.wQKV, d.wZ = r.mat(hidden, d.convDim), r.mat(hidden, vd*heads)
	d.wA, d.wB = r.mat(hidden, heads), r.mat(hidden, heads)
	d.wOut = r.mat(vd*heads, hidden)
	d.conv, d.aLog, d.dtBias, d.norm = r.vec(d.convDim*convK), r.vec(heads), r.vec(heads), r.vec(vd)

	m := &qwen{blocks: []qblock{{delta: d, dstate: d.newState()}}}
	scratch := newDeltaScratch(d, hidden)
	x := r.vec(hidden)
	d.step(m.blocks[0].dstate, x, scratch)
	snap := snapshotDelta(m)
	want := d.step(m.blocks[0].dstate, x, scratch)
	want = append([]float32(nil), want...)

	// Move the state on, then put it back and take the same step again.
	for i := 0; i < 3; i++ {
		d.step(m.blocks[0].dstate, r.vec(hidden), scratch)
	}
	restoreDelta(m, snap)
	got := d.step(m.blocks[0].dstate, x, scratch)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("after restore out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
