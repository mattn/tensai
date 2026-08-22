// Command wgpu exercises the experimental WebGPU backend: it reports the
// adapter wgpu-native picked, checks a GPU MatMul against the CPU one, and
// times both. With -sweep it walks a ladder of sizes instead and prints
// where the GPU overtakes the CPU kernel.
//
// The CPU side is the same dotRows kernel the rest of the package uses, so
// building twice pins down all three implementations:
//
//	GOEXPERIMENT=nosimd go build -tags wgpu -o wgpu-nosimd ./_example/wgpu
//	GOEXPERIMENT=simd   go build -tags wgpu -o wgpu-simd   ./_example/wgpu
//
// the first binary's cpu column is the portable Go kernel and the second's
// is the AVX2 one. gpu+xfer includes input uploads; resident reuses inputs
// already on the device. Both GPU columns include the final download.
//
// The GPU path only exists in builds with -tags wgpu on linux, darwin, or
// windows; anywhere else OpenGPU fails cleanly and this command explains
// why. The wgpu-native shared library is loaded at runtime, so point
// TENSAI_WGPU_LIB at it when it is not on the loader's path:
//
//	TENSAI_WGPU_LIB=$PWD/wgpu/lib/libwgpu_native.so \
//	    go run -tags wgpu ./_example/wgpu
//
// On Windows the library is wgpu_native.dll:
//
//	set TENSAI_WGPU_LIB=%CD%\wgpu\lib\wgpu_native.dll
//	go run -tags wgpu ./_example/wgpu
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	tensai "github.com/mattn/tensai"
)

// Power preferences are a hint: with a single adapter you always get that
// one, so this only matters on machines with an iGPU and a dGPU.
var powers = map[string]tensai.GPUPower{
	"default": tensai.GPUDefault,
	"low":     tensai.GPULowPower,
	"high":    tensai.GPUHighPerformance,
}

// shape is one (batch, m, k, n) point on the sweep ladder.
type shape struct {
	label          string
	batch, m, k, n int
}

// ladder climbs from products the transfer swamps to ones big enough for
// the GPU to win, with the two MNIST layer shapes thrown in because they
// are the sizes a small model actually asks for.
var ladder = []shape{
	{"mnist dense", 1, 100, 784, 128},
	{"mnist conv2", 1, 19600, 72, 16},
	{"tiny", 1, 128, 128, 128},
	{"small", 1, 512, 512, 512},
	{"medium", 8, 512, 512, 512},
	{"large", 32, 512, 512, 512},
	{"huge", 64, 512, 512, 512},
}

func randTensor(rng *rand.Rand, dims ...int) *tensai.Tensor {
	t := tensai.NewTensor(dims...)
	for i := range t.Data {
		t.Data[i] = float32(rng.NormFloat64())
	}
	return t
}

// maxDiff is the largest absolute elementwise difference. Both sides sum
// the k axis in f32, and whether they do it in the same order is up to the
// driver, so close agreement is the promise — bit-exact equality happens
// but is not one.
func maxDiff(a, b *tensai.Tensor) float64 {
	var worst float64
	for i := range a.Data {
		if d := math.Abs(float64(a.Data[i] - b.Data[i])); d > worst {
			worst = d
		}
	}
	return worst
}

// minRun is how long each timing window has to last. Windows' clock ticks
// at about half a millisecond, so a small product has to be repeated until
// the total is comfortably above that to time as anything but zero.
const minRun = 20 * time.Millisecond

// timeOp runs fn back to back until minRun has passed and returns the
// average per call.
func timeOp(fn func() error) (time.Duration, error) {
	start := time.Now()
	for iters := 1; ; iters++ {
		if err := fn(); err != nil {
			return 0, err
		}
		if elapsed := time.Since(start); elapsed >= minRun {
			return elapsed / time.Duration(iters), nil
		}
	}
}

// measure times the convenient upload/matmul/download path, the usual
// resident-input GPU path, and the CPU. Resident timing still downloads the
// result on every call, both to synchronize the GPU and to model inference
// that consumes its final output on the host; only repeated input uploads are
// removed. The first calls warm the driver's caches and are not timed.
func measure(gpu *tensai.GPU, s shape, reps int) (transferTime, residentTime, cpuTime time.Duration, diff float64, err error) {
	rng := rand.New(rand.NewSource(1))
	a := randTensor(rng, s.batch, s.m, s.k)
	w := randTensor(rng, s.k, s.n)

	if _, err := gpu.MatMul(a, w); err != nil {
		return 0, 0, 0, 0, err
	}
	ga, err := gpu.Upload(a)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer ga.Free()
	gw, err := gpu.Upload(w)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer gw.Free()

	var transferGot, residentGot, want *tensai.Tensor
	residentOp := func() error {
		out, e := ga.MatMul(gw)
		if e != nil {
			return e
		}
		defer out.Free()
		residentGot, e = out.Download()
		return e
	}
	if err := residentOp(); err != nil {
		return 0, 0, 0, 0, err
	}

	transferTime, residentTime, cpuTime = time.Hour, time.Hour, time.Hour
	for i := 0; i < reps; i++ {
		d, err := timeOp(func() error {
			var e error
			transferGot, e = gpu.MatMul(a, w)
			return e
		})
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if d < transferTime {
			transferTime = d
		}

		d, err = timeOp(residentOp)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if d < residentTime {
			residentTime = d
		}

		d, err = timeOp(func() error {
			var e error
			want, e = tensai.MatMul(a, w)
			return e
		})
		if err != nil {
			return 0, 0, 0, 0, err
		}
		if d < cpuTime {
			cpuTime = d
		}
	}
	diff = math.Max(maxDiff(transferGot, want), maxDiff(residentGot, want))
	return transferTime, residentTime, cpuTime, diff, nil
}

// mflop counts the multiply-add pairs in one product, in millions.
func mflop(s shape) float64 {
	return 2 * float64(s.batch) * float64(s.m) * float64(s.k) * float64(s.n) / 1e6
}

func sweep(gpu *tensai.GPU, reps int) {
	fmt.Printf("%-12s %-18s %10s %10s %10s %10s %9s   %s\n",
		"", "shape", "MFLOP", "gpu+xfer", "resident", "cpu", "res/cpu", "max diff")
	crossed := false
	for _, s := range ladder {
		transferTime, residentTime, cpuTime, diff, err := measure(gpu, s, reps)
		if err != nil {
			fmt.Printf("%-12s %-18s %10s\n", s.label,
				fmt.Sprintf("%dx%dx%d@%dx%d", s.batch, s.m, s.k, s.k, s.n), "failed: "+err.Error())
			continue
		}
		ratio := float64(cpuTime) / float64(residentTime)
		mark := ""
		if ratio >= 1 && !crossed {
			crossed = true
			mark = "  <- gpu takes the lead here"
		}
		fmt.Printf("%-12s %-18s %10.1f %10s %10s %10s %8.2fx   %.1e%s\n",
			s.label,
			fmt.Sprintf("%dx%dx%d@%dx%d", s.batch, s.m, s.k, s.k, s.n),
			mflop(s),
			transferTime.Round(time.Microsecond),
			residentTime.Round(time.Microsecond),
			cpuTime.Round(time.Microsecond),
			ratio, diff, mark)
	}
	if !crossed {
		fmt.Println("\nthe CPU kernel won every size: this machine's GPU never")
		fmt.Println("earns back the dispatch and final readback at these shapes")
	}
}

func main() {
	power := flag.String("power", "default", "adapter preference: default, low, high")
	doSweep := flag.Bool("sweep", false, "time a ladder of sizes instead of one product")
	reps := flag.Int("reps", 3, "timing windows per size; the fastest is reported")
	batch := flag.Int("batch", 32, "batch size")
	m := flag.Int("m", 512, "rows of a")
	k := flag.Int("k", 512, "shared dimension")
	n := flag.Int("n", 512, "columns of b")
	flag.Parse()

	pref, ok := powers[*power]
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown -power %q (want default, low, or high)\n", *power)
		os.Exit(2)
	}

	gpu, err := tensai.OpenGPU(pref)
	if err != nil {
		fmt.Fprintln(os.Stderr, "no GPU backend:", err)
		fmt.Fprintln(os.Stderr, "hint: build with -tags wgpu (linux, darwin, or windows) and")
		fmt.Fprintln(os.Stderr, "      install wgpu-native v22.1.0.5, or point TENSAI_WGPU_LIB at it")
		os.Exit(1)
	}
	defer gpu.Close()
	fmt.Printf("adapter: %s\n\n", gpu.Name())

	// A product small enough to read: [[1,2],[3,4]] squared is
	// [[7,10],[15,22]], computed by the WGSL shader.
	x, err := tensai.NewTensorFromSlice([]float32{1, 2, 3, 4}, 2, 2)
	if err != nil {
		panic(err)
	}
	sq, err := gpu.MatMul(x, x)
	if err != nil {
		panic(err)
	}
	fmt.Printf("x @ x = [[%g %g] [%g %g]]\n\n",
		sq.At(0, 0), sq.At(0, 1), sq.At(1, 0), sq.At(1, 1))

	if *doSweep {
		sweep(gpu, *reps)
		return
	}

	// One product, with the shape from the flags. Broadcasting matches the
	// CPU MatMul: a 2-D weight stretches across the batch axis, so
	// (batch, m, k) @ (k, n) -> (batch, m, n) is a single dispatch with one
	// upload of the weight.
	s := shape{"", *batch, *m, *k, *n}
	transferTime, residentTime, cpuTime, diff, err := measure(gpu, s, *reps)
	if err != nil {
		panic(err)
	}
	fmt.Printf("a[%d %d %d] @ w[%d %d] = out[%d %d %d] (%.1f MFLOP)\n",
		s.batch, s.m, s.k, s.k, s.n, s.batch, s.m, s.n, mflop(s))
	fmt.Printf("  gpu + input upload: %v\n", transferTime)
	fmt.Printf("  gpu, resident inputs: %v\n", residentTime)
	fmt.Printf("  cpu: %v\n", cpuTime)
	fmt.Printf("  max |gpu - cpu|: %.2e\n", diff)

	// The resident path still reads the final product back, so the GPU only
	// wins once the arithmetic outgrows dispatch and readback. Run -sweep to
	// compare that path with the convenient per-call upload path.
	if ratio := float64(cpuTime) / float64(residentTime); ratio >= 1 {
		fmt.Printf("\ngpu is %.2fx faster than the CPU kernel\n", ratio)
	} else {
		fmt.Printf("\ngpu with resident inputs is %.2fx slower than the CPU kernel:\n"+
			"this product is too small to pay for dispatch and final readback\n", 1/ratio)
	}
}
