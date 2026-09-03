//go:build !(goexperiment.simd && (amd64 || (arm64 && go1.27)))

// Package simd wraps the experimental simd/archsimd load/store calls whose
// spellings changed between Go releases, so the kernels can target one set
// of names. On builds without GOEXPERIMENT=simd this stub keeps the package
// compiling; nothing imports it there.
package simd

// HasAVX2 and HasNEON are always false without the simd experiment.
const (
	HasAVX2 = false
	HasNEON = false
)
