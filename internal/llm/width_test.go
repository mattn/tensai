package llm

import "testing"

func TestPickBits(t *testing.T) {
	const gb = 1 << 30
	for _, c := range []struct {
		stored string
		params int64
		avail  int64
		want   int
	}{
		{"Q4_K", 7e9, 0, 4}, // int4 blocks stay int4, whatever the memory
		{"Q4_0", 1e9, 64 * gb, 4},
		{"MXFP4", 20e9, 64 * gb, 4},
		{"PTQ1_0", 27e9, 15 * gb, 8}, // ternary ignores the width
		{"Q8_0", 1e9, 15 * gb, 8},
		{"Q8_0", 7e9, 15 * gb, 8},  // 8.75GB + 0.5GB fits 15GB
		{"Q8_0", 13e9, 15 * gb, 4}, // 16.8GB does not
		{"Q6_K", 7e9, 8 * gb, 4},
		{"Q6_K", 7e9, 0, 8}, // memory unknown: keep the precision
		{"", 0.5e9, 4 * gb, 8},
		{"", 7e9, 8 * gb, 4},
		{"F16", 3e9, 15 * gb, 8},
	} {
		got, why := pickBits(c.stored, c.params, c.avail)
		if got != c.want {
			t.Errorf("pickBits(%q, %.1fB, %dGB) = %d (%s), want %d", c.stored, float64(c.params)/1e9, c.avail/gb, got, why, c.want)
		}
	}
}
