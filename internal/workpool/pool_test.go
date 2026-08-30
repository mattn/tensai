package workpool

import (
	"sync"
	"sync/atomic"
	"testing"
)

func TestRunCoversRange(t *testing.T) {
	for _, tc := range []struct{ n, align int }{
		{1003, 8}, {16, 8}, {1, 1}, {4864, 32}, {7, 16},
	} {
		hits := make([]int32, tc.n)
		Run(tc.n, tc.align, func(lo, hi int) {
			if lo < 0 || hi > tc.n || lo >= hi {
				t.Errorf("n=%d align=%d: bad chunk [%d,%d)", tc.n, tc.align, lo, hi)
				return
			}
			if lo%tc.align != 0 {
				t.Errorf("n=%d align=%d: chunk start %d not aligned", tc.n, tc.align, lo)
			}
			for i := lo; i < hi; i++ {
				atomic.AddInt32(&hits[i], 1)
			}
		})
		for i, h := range hits {
			if h != 1 {
				t.Fatalf("n=%d align=%d: index %d covered %d times", tc.n, tc.align, i, h)
			}
		}
	}
}

// Concurrent submitters contend for the single op slot; the losers must
// still cover their whole range through the serial fallback.
func TestRunConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for rep := 0; rep < 50; rep++ {
				var sum atomic.Int64
				Run(999, 8, func(lo, hi int) {
					for i := lo; i < hi; i++ {
						sum.Add(int64(i))
					}
				})
				if got := sum.Load(); got != 999*998/2 {
					t.Errorf("sum = %d, want %d", got, 999*998/2)
					return
				}
			}
		}()
	}
	wg.Wait()
}
