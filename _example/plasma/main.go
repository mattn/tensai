// Command plasma renders a demoscene-style plasma effect in the terminal —
// except the plasma function is a neural network. A small randomly weighted
// MLP (a CPPN: x, y, radius, and two time-animated waves in; RGB out) is
// evaluated for every pixel of every frame as one big tensai batch, drawn
// with 24-bit ANSI colors and half-block characters.
//
// The whole frame is a single (width*height*2 x 5) matrix flowing through
// Dense/Tanh layers, so the "nn" time in the status line is a live measure
// of the matmul kernels: build with GOEXPERIMENT=simd and watch it drop.
//
//	go run ./_example/plasma
//	go run ./_example/plasma -w 120 -h 40 -seed 11
//	go run ./_example/plasma -frames 300   # exit after 300 frames
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"strconv"
	"time"

	tensai "github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
	"github.com/mattn/tensai/loss"
	"github.com/mattn/tensai/model"
	"github.com/mattn/tensai/optim"
)

const features = 5 // x, y, r, sin wave, cos wave

// buildCPPN assembles the pattern network. The Dense layers are kept so
// their weights can be re-randomized: Glorot initialization is tuned for
// trainability, but a CPPN wants big weights so tanh folds the coordinate
// space into rich patterns.
func buildCPPN(seed int64, gain float32) *model.Sequential {
	denses := []*layer.Dense{
		layer.NewDense(32),
		layer.NewDense(32),
		layer.NewDense(32),
		layer.NewDense(3),
	}
	model := model.NewSequential()
	model.Add(denses[0])
	model.Add(&layer.Tanh{})
	model.Add(denses[1])
	model.Add(&layer.Tanh{})
	model.Add(denses[2])
	model.Add(&layer.Tanh{})
	model.Add(denses[3])
	model.Add(&layer.Sigmoid{})
	if err := model.Compile(features, loss.MeanSquaredError{}, optim.NewSGD(0, 0)); err != nil {
		panic(err)
	}
	rng := rand.New(rand.NewSource(seed))
	for _, d := range denses {
		w, _ := d.Params()
		scale := gain / float32(math.Sqrt(float64(w.Rows)))
		for i := range w.Data {
			w.Data[i] = float32(rng.NormFloat64()) * scale
		}
	}
	return model
}

func main() {
	width := flag.Int("w", 100, "width in terminal cells")
	height := flag.Int("h", 40, "height in terminal cells (2 pixels per cell)")
	seed := flag.Int64("seed", 5, "weight seed: every seed is a different effect")
	gain := flag.Float64("gain", 6.0, "weight gain; higher = busier patterns")
	frames := flag.Int("frames", 0, "stop after this many frames (0 = run until Ctrl-C)")
	flag.Parse()

	W, H := *width, *height
	px := W * H * 2 // half blocks: two pixels per cell

	model := buildCPPN(*seed, float32(*gain))

	// Static per-pixel coordinates; the two wave features animate per frame.
	in := tensai.NewMatrix(px, features)
	aspect := float32(W) / float32(2*H)
	xs := make([]float32, px)
	ys := make([]float32, px)
	for y := 0; y < 2*H; y++ {
		for x := 0; x < W; x++ {
			p := y*W + x
			fx := (float32(x)/float32(W-1)*2 - 1) * aspect
			fy := float32(y)/float32(2*H-1)*2 - 1
			xs[p], ys[p] = fx, fy
			in.Data[p*features+0] = fx
			in.Data[p*features+1] = fy
			in.Data[p*features+2] = float32(math.Sqrt(float64(fx*fx + fy*fy)))
		}
	}

	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt)
	fmt.Print("\x1b[2J\x1b[?25l") // clear screen, hide cursor
	defer fmt.Print("\x1b[0m\x1b[?25h\n")

	buf := make([]byte, 0, px*40)
	start := time.Now()
	var fps, nnMillis float64
	for frame := 0; *frames == 0 || frame < *frames; frame++ {
		select {
		case <-interrupt:
			return
		default:
		}
		frameStart := time.Now()
		t := float64(time.Since(start).Seconds())

		// Animate: two travelling waves coupled to the coordinates.
		for p := 0; p < px; p++ {
			in.Data[p*features+3] = float32(math.Sin(2.0*float64(xs[p]) + 1.3*t))
			in.Data[p*features+4] = float32(math.Cos(2.0*float64(ys[p]) - 0.9*t))
		}

		nnStart := time.Now()
		out, err := model.Predict(in)
		if err != nil {
			panic(err)
		}
		nnMillis = 0.9*nnMillis + 0.1*float64(time.Since(nnStart).Microseconds())/1000

		buf = buf[:0]
		buf = append(buf, "\x1b[H"...)
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				buf = appendColor(buf, out, (2*y)*W+x, 38)   // foreground: top pixel
				buf = appendColor(buf, out, (2*y+1)*W+x, 48) // background: bottom pixel
				buf = append(buf, "▀"...)
			}
			buf = append(buf, "\x1b[0m\n"...)
		}
		if d := time.Since(frameStart).Seconds(); d > 0 {
			fps = 0.9*fps + 0.1/d
		}
		buf = append(buf, fmt.Sprintf("\x1b[0m tensai plasma  %dx%d px  nn %5.1f ms  %5.1f fps  seed=%d  (Ctrl-C to quit)",
			W, 2*H, nnMillis, fps, *seed)...)
		os.Stdout.Write(buf)
	}
}

// appendColor appends an ANSI 24-bit color escape (mode 38 = foreground,
// 48 = background) for one prediction row.
func appendColor(buf []byte, m *tensai.Matrix, row, mode int) []byte {
	buf = append(buf, "\x1b["...)
	buf = strconv.AppendInt(buf, int64(mode), 10)
	buf = append(buf, ";2;"...)
	for c := 0; c < 3; c++ {
		v := m.At(row, c)
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		buf = strconv.AppendInt(buf, int64(v*255), 10)
		if c < 2 {
			buf = append(buf, ';')
		}
	}
	return append(buf, 'm')
}
