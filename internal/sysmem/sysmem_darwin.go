//go:build darwin

package sysmem

import (
	"encoding/binary"
	"syscall"
)

// Available is the machine's memory: macOS reports no available figure
// the way Linux does, and its page cache yields readily, so the total
// is the honest ceiling.
func Available() int64 {
	s, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	// Sysctl returns the raw bytes as a string, trailing NUL trimmed;
	// pad back to the eight the value has.
	b := []byte(s)
	for len(b) < 8 {
		b = append(b, 0)
	}
	return int64(binary.LittleEndian.Uint64(b[:8]))
}
