//go:build goexperiment.simd && arm64 && go1.27

package quant

import (
	"encoding/binary"
	"os"
	"runtime"
)

// sdotTile32 accumulates a 32-column tile: acc holds eight int32 vectors,
// tile the quad-major weights, xu the activation row in its offset-binary
// form. Implemented in sdot_arm64.s.
//
//go:noescape
func sdotTile32(acc *int32, tile *int8, xu *uint8, quads int)

// hasDotProd reports FEAT_DotProd, the ARMv8.2 extension whose SDOT the
// tile kernel needs. It is mandatory from ARMv8.4, so every Apple core
// Go runs on has it, but an older v8.0 part (a Cortex-A72, say) does not
// and has to keep the widening path.
var hasDotProd = detectDotProd()

func detectDotProd() bool {
	switch runtime.GOOS {
	case "darwin", "ios":
		// Every arm64 core Apple has shipped implements it, and Go's
		// darwin/arm64 port runs on nothing else.
		return true
	case "linux", "android":
		return auxvHasDotProd()
	}
	// Elsewhere the answer is unknown, and the widening kernels are
	// correct everywhere.
	return false
}

// auxvHasDotProd reads the kernel's hardware capability word out of the
// process's own auxiliary vector. x/sys would spell this in one call,
// and internal/cpu already knows the answer, but neither is reachable
// from here without a dependency.
func auxvHasDotProd() bool {
	const (
		atHWCap      = 16      // AT_HWCAP
		hwcapASIMDDP = 1 << 20 // HWCAP_ASIMDDP
		entrySize    = 16      // two 64-bit words per entry
	)
	b, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	for i := 0; i+entrySize <= len(b); i += entrySize {
		tag := binary.LittleEndian.Uint64(b[i:])
		val := binary.LittleEndian.Uint64(b[i+8:])
		if tag == 0 {
			break
		}
		if tag == atHWCap {
			return val&hwcapASIMDDP != 0
		}
	}
	return false
}
