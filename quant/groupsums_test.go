package quant

import (
	"math/rand/v2"
	"testing"
)

// groupSumsRef is the index-clamping loop groupSums replaced, kept as the
// reference it answers to.
func groupSumsRef(gsum []int32, xu []uint8, grp int) {
	for i := range gsum {
		gsum[i] = 0
	}
	for i, u := range xu {
		gsum[min(i/grp, len(gsum)-1)] += int32(u) - 64
	}
}

func TestGroupSums(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 0))
	// Row counts that straddle the group and quad boundaries: exact
	// multiples, one over, one under, and the shapes the models actually
	// use (896 and 4864 for Qwen2.5-0.5B's projections).
	for _, rows := range []int{1, 3, 4, 7, 63, 64, 65, 100, 127, 128, 129, 130, 255, 896, 1152, 4864, 4865} {
		for _, grp := range []int{q4Group, 32, 128} {
			padded := (rows + 3) &^ 3
			xu := make([]uint8, padded)
			for i := range xu {
				xu[i] = uint8(rng.UintN(256))
			}
			for i := rows; i < padded; i++ {
				xu[i] = 64 // the pad value quantizeActs writes
			}
			n := (rows + grp - 1) / grp
			got := make([]int32, n)
			want := make([]int32, n)
			groupSums(got, xu, grp)
			groupSumsRef(want, xu, grp)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("rows=%d grp=%d group %d: got %d want %d", rows, grp, i, got[i], want[i])
				}
			}
			// The sums must also account for every element exactly once.
			var total, sum int32
			for _, u := range xu {
				total += int32(u) - 64
			}
			for _, v := range got {
				sum += v
			}
			if sum != total {
				t.Fatalf("rows=%d grp=%d: sums total %d, want %d", rows, grp, sum, total)
			}
		}
	}
}
