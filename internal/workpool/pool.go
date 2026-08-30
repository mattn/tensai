// Package workpool runs decode-time parallel work on resident workers.
//
// A decode step is ~100 short parallel regions (~200us each) per token,
// and spawning goroutines puts a sleep/wake on every one of them: the
// wake cost, not memory access, is what keeps the cycled-weight
// bandwidth at 62% of a single sweep (see PERF-INVESTIGATION.md). ggml
// hides it with a resident thread pool that spins on PAUSE between
// regions. Earlier Go attempts substituted Gosched or a bare busy loop
// for PAUSE and lost — a yielded P gets rescheduled, and a bare spin
// starves the sibling hyperthread. This pool spins on the real PAUSE
// instruction, briefly, then parks: within a token the next region
// arrives in microseconds, so the spin window almost always absorbs it,
// and the park bounds the burn between tokens and after decode ends.
package workpool

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type op struct {
	body   func(lo, hi int)
	n      int
	chunk  int
	cursor atomic.Int64
	left   atomic.Int64
	chunks int64
}

var (
	setupOnce sync.Once
	workers   int
	seq       atomic.Uint64
	cur       atomic.Pointer[op]
	parked    atomic.Int64
	wake      chan struct{}
	busy      atomic.Bool
)

// spinFor is how long a worker spins on PAUSE waiting for the next
// region before parking. Regions inside one token are microseconds
// apart; sampling between tokens is longer and should park. Swept over
// 5us-2ms: shorter loses regions to parking mid-token, longer burns the
// hyperthread siblings through the serial stretches for nothing.
const spinFor = 20 * time.Microsecond

func setup() {
	workers = runtime.GOMAXPROCS(0)
	if workers < 2 {
		workers = 0
		return
	}
	wake = make(chan struct{}, workers)
	for i := 0; i < workers-1; i++ {
		go worker()
	}
}

func worker() {
	var last uint64
	for {
		if s := seq.Load(); s != last {
			last = s
			if o := cur.Load(); o != nil {
				runOp(o)
			}
			continue
		}
		deadline := time.Now().Add(spinFor)
		fresh := false
		for !fresh {
			for i := 0; i < 256; i++ {
				if seq.Load() != last {
					fresh = true
					break
				}
				pause()
			}
			if !fresh && !time.Now().Before(deadline) {
				break
			}
		}
		if fresh {
			continue
		}
		// Parking publishes parked before re-reading seq, and Run
		// publishes seq before reading parked (both sequentially
		// consistent), so a submit can miss this worker's increment only
		// if the worker saw the new seq — no lost wakeup either way.
		parked.Add(1)
		if seq.Load() != last {
			parked.Add(-1)
			continue
		}
		<-wake
		parked.Add(-1)
	}
}

func runOp(o *op) {
	for {
		c := o.cursor.Add(1) - 1
		if c >= o.chunks {
			return
		}
		lo := int(c) * o.chunk
		hi := min(lo+o.chunk, o.n)
		o.body(lo, hi)
		o.left.Add(-1)
	}
}

// Run splits [0, n) into chunks whose bounds are multiples of align and
// runs body over them on the resident workers, the caller included, and
// returns when every chunk has finished. When the pool is disabled or
// already mid-op the caller just runs the whole range itself.
func Run(n, align int, body func(lo, hi int)) {
	setupOnce.Do(setup)
	if workers == 0 || !busy.CompareAndSwap(false, true) {
		body(0, n)
		return
	}
	chunk := ((n+workers-1)/workers + align - 1) &^ (align - 1)
	if chunk <= 0 {
		chunk = align
	}
	o := &op{body: body, n: n, chunk: chunk}
	o.chunks = int64((n + chunk - 1) / chunk)
	o.left.Store(o.chunks)
	cur.Store(o)
	seq.Add(1)
	if p := parked.Load(); p > 0 {
		for i := int64(0); i < p; i++ {
			select {
			case wake <- struct{}{}:
			default:
			}
		}
	}
	runOp(o)
	// The last chunks may still be running on workers; they are at most
	// one region long, so waiting is short. The occasional yield keeps a
	// preempted worker from deadlocking against a full complement of
	// spinning peers.
	for i := 0; o.left.Load() != 0; i++ {
		pause()
		if i&1023 == 1023 {
			runtime.Gosched()
		}
	}
	busy.Store(false)
}
