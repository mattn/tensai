// Package qwenaudio runs the audio side of Qwen2-Audio: a WAV file becomes
// Whisper's log-mel spectrogram, Whisper-large-v3's encoder turns that
// into one vector per 40ms, and a projection puts those vectors in the
// language model's embedding space, where they stand in for the prompt's
// audio placeholder. The language model itself is internal/llm's.
package qwenaudio

import (
	"math"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/workpool"
)

// Whisper's front end, as the checkpoint's preprocessor_config.json
// describes it: thirty seconds at 16kHz, a 400-sample window every 160
// samples, 128 mel bands.
const (
	SampleRate = 16000
	melSamples = 30 * SampleRate
	melFrames  = melSamples / hop // 3000
	nFFT       = 400
	hop        = 160
	nBins      = nFFT/2 + 1 // 201
	nMels      = 128
)

// LogMel is Whisper's feature extractor. The samples are padded (or cut)
// to thirty seconds and turned into a (128, 3000) log-mel spectrogram;
// frames is how many of its columns came from the audio rather than the
// padding, which is what decides how many tokens the audio becomes.
func LogMel(samples []float32) (mel *tensai.Matrix, frames int) {
	n := min(len(samples), melSamples)
	frames = (n + hop - 1) / hop

	// The STFT centres each window on its frame, reflecting the padded
	// signal at both ends.
	padded := make([]float64, melSamples+nFFT)
	for i := range melSamples {
		if i < n {
			padded[i+nFFT/2] = float64(samples[i])
		}
	}
	for i := 1; i <= nFFT/2; i++ {
		padded[nFFT/2-i] = padded[nFFT/2+i]
		padded[nFFT/2+melSamples-1+i] = padded[nFFT/2+melSamples-1-i]
	}

	// A frame whose window lies wholly in the zero padding has no power
	// in any band, and the floor below is what it would come to.
	live := min(melFrames, (n+nFFT/2)/hop+1)
	logs := make([]float64, nMels*melFrames)
	for i := range logs {
		logs[i] = math.Log10(melFloor)
	}
	window, cos, sin, filters := melTables()
	workpool.Run(live, 1, func(lo, hi int) {
		frame := make([]float64, nFFT)
		power := make([]float64, nBins)
		for t := lo; t < hi; t++ {
			for i := range frame {
				frame[i] = padded[t*hop+i] * window[i]
			}
			for k := range nBins {
				var re, im float64
				for i, x := range frame {
					re += x * cos[(k*i)%nFFT]
					im -= x * sin[(k*i)%nFFT]
				}
				power[k] = re*re + im*im
			}
			for m := range nMels {
				var e float64
				for k, p := range power {
					e += filters[m*nBins+k] * p
				}
				logs[m*melFrames+t] = math.Log10(max(e, melFloor))
			}
		}
	})

	top := math.Inf(-1)
	for _, v := range logs {
		top = max(top, v)
	}
	mel = tensai.NewMatrix(nMels, melFrames)
	for i, v := range logs {
		mel.Data[i] = tensai.Float((max(v, top-8) + 4) / 4)
	}
	return mel, frames
}

// melFloor keeps a silent band's logarithm finite.
const melFloor = 1e-10

// melTables returns the periodic Hann window, the DFT's twiddles, and
// the slaney-normalized mel filter bank, (128, 201) row by row.
func melTables() (window, cos, sin, filters []float64) {
	window = make([]float64, nFFT)
	cos = make([]float64, nFFT)
	sin = make([]float64, nFFT)
	for i := range nFFT {
		a := 2 * math.Pi * float64(i) / nFFT
		window[i] = 0.5 - 0.5*math.Cos(a)
		cos[i], sin[i] = math.Cos(a), math.Sin(a)
	}
	// Band edges are spaced evenly on the slaney mel scale from 0 to
	// the Nyquist frequency; each filter is a triangle over its two
	// neighbours' centres, scaled to unit area.
	edges := make([]float64, nMels+2)
	top := hzToMel(SampleRate / 2)
	for i := range edges {
		edges[i] = melToHz(top * float64(i) / float64(nMels+1))
	}
	filters = make([]float64, nMels*nBins)
	for m := range nMels {
		norm := 2 / (edges[m+2] - edges[m])
		for k := range nBins {
			f := float64(SampleRate/2) * float64(k) / float64(nBins-1)
			down := (f - edges[m]) / (edges[m+1] - edges[m])
			up := (edges[m+2] - f) / (edges[m+2] - edges[m+1])
			filters[m*nBins+k] = max(0, min(down, up)) * norm
		}
	}
	return window, cos, sin, filters
}

// The slaney mel scale is linear below 1kHz and logarithmic above.
const (
	minLogHz  = 1000.0
	minLogMel = 15.0
)

var logStep = 27 / math.Log(6.4)

func hzToMel(hz float64) float64 {
	if hz < minLogHz {
		return 3 * hz / 200
	}
	return minLogMel + math.Log(hz/minLogHz)*logStep
}

func melToHz(mel float64) float64 {
	if mel < minLogMel {
		return 200 * mel / 3
	}
	return minLogHz * math.Exp((mel-minLogMel)/logStep)
}
