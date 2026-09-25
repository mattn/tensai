package qwenaudio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// ReadWAV reads a RIFF WAVE file as mono samples in [-1, 1) at 16kHz:
// channels are averaged and any other rate is resampled. It takes 8, 16,
// 24 and 32-bit PCM and 32 or 64-bit float, which covers what recorders
// and ffmpeg write; anything compressed wants converting first.
func ReadWAV(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	samples, rate, err := decodeWAV(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return Resample(samples, rate, SampleRate), nil
}

// DecodeWAV is ReadWAV for a file already in memory, the way an API
// request carries one.
func DecodeWAV(data []byte) ([]float32, error) {
	samples, rate, err := decodeWAV(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return Resample(samples, rate, SampleRate), nil
}

func decodeWAV(r io.Reader) ([]float32, int, error) {
	var head [12]byte
	if _, err := io.ReadFull(r, head[:]); err != nil {
		return nil, 0, errors.New("not a WAV file")
	}
	if string(head[:4]) != "RIFF" || string(head[8:]) != "WAVE" {
		return nil, 0, errors.New("not a WAV file")
	}
	var (
		format, channels, bits int
		rate                   int
		haveFmt                bool
	)
	for {
		var ch [8]byte
		if _, err := io.ReadFull(r, ch[:]); err != nil {
			return nil, 0, errors.New("no data chunk")
		}
		size := int64(binary.LittleEndian.Uint32(ch[4:]))
		switch string(ch[:4]) {
		case "fmt ":
			b := make([]byte, size+size%2)
			if _, err := io.ReadFull(r, b); err != nil || size < 16 {
				return nil, 0, errors.New("short fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(b))
			channels = int(binary.LittleEndian.Uint16(b[2:]))
			rate = int(binary.LittleEndian.Uint32(b[4:]))
			bits = int(binary.LittleEndian.Uint16(b[14:]))
			// WAVE_FORMAT_EXTENSIBLE names the real format in the
			// first two bytes of its sub-format GUID.
			if format == 0xfffe && size >= 26 {
				format = int(binary.LittleEndian.Uint16(b[24:]))
			}
			haveFmt = true
		case "data":
			if !haveFmt {
				return nil, 0, errors.New("data chunk before fmt")
			}
			if channels < 1 || rate < 1 {
				return nil, 0, fmt.Errorf("%d channels at %dHz", channels, rate)
			}
			// A stream still being written says 0 or 0xffffffff;
			// either way the chunk runs to the end of the file.
			var data []byte
			var err error
			if size == 0 || size == 0xffffffff {
				data, err = io.ReadAll(r)
			} else {
				data = make([]byte, size)
				_, err = io.ReadFull(r, data)
			}
			if err != nil {
				return nil, 0, fmt.Errorf("reading samples: %w", err)
			}
			s, err := pcm(data, format, bits, channels)
			return s, rate, err
		default:
			if _, err := io.CopyN(io.Discard, r, size+size%2); err != nil {
				return nil, 0, errors.New("no data chunk")
			}
		}
	}
}

// pcm decodes interleaved frames and averages their channels.
func pcm(data []byte, format, bits, channels int) ([]float32, error) {
	var read func([]byte) float64
	switch {
	case format == 1 && bits == 8:
		read = func(b []byte) float64 { return (float64(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		read = func(b []byte) float64 { return float64(int16(binary.LittleEndian.Uint16(b))) / (1 << 15) }
	case format == 1 && bits == 24:
		read = func(b []byte) float64 {
			return float64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / (1 << 23)
		}
	case format == 1 && bits == 32:
		read = func(b []byte) float64 { return float64(int32(binary.LittleEndian.Uint32(b))) / (1 << 31) }
	case format == 3 && bits == 32:
		read = func(b []byte) float64 { return float64(math.Float32frombits(binary.LittleEndian.Uint32(b))) }
	case format == 3 && bits == 64:
		read = func(b []byte) float64 { return math.Float64frombits(binary.LittleEndian.Uint64(b)) }
	default:
		return nil, fmt.Errorf("unsupported sample format %d at %d bits; convert it to PCM first", format, bits)
	}
	width := bits / 8 * channels
	out := make([]float32, len(data)/width)
	for i := range out {
		var sum float64
		for c := range channels {
			sum += read(data[i*width+c*bits/8:])
		}
		out[i] = float32(sum / float64(channels))
	}
	return out, nil
}

// Resample converts samples between rates with a windowed-sinc filter
// whose cutoff sits below the lower rate's Nyquist frequency, so
// downsampling does not fold the discarded band back in.
func Resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	const zeros = 16 // sinc lobes on each side of the centre
	ratio := float64(to) / float64(from)
	cutoff := 0.95 * min(1, ratio)
	half := int(math.Ceil(zeros / cutoff))
	out := make([]float32, int(math.Ceil(float64(len(x))*ratio)))
	for i := range out {
		centre := float64(i) / ratio
		c := int(math.Floor(centre))
		var sum, weight float64
		for j := c - half + 1; j <= c+half; j++ {
			if j < 0 || j >= len(x) {
				continue
			}
			d := centre - float64(j)
			w := cutoff * sinc(cutoff*d) * blackman(d/float64(half))
			sum += w * float64(x[j])
			weight += w
		}
		if weight != 0 {
			// Normalizing by the taps actually summed keeps the edges,
			// where the kernel is cut short, at the right level.
			out[i] = float32(sum / weight)
		}
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// blackman is the Blackman window over [-1, 1].
func blackman(t float64) float64 {
	if t <= -1 || t >= 1 {
		return 0
	}
	a := math.Pi * (t + 1)
	return 0.42 - 0.5*math.Cos(a) + 0.08*math.Cos(2*a)
}
