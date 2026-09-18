//go:build windows

package sysmem

import (
	"syscall"
	"unsafe"
)

// Available is GlobalMemoryStatusEx's available physical memory.
func Available() int64 {
	var st struct {
		Length               uint32
		MemoryLoad           uint32
		TotalPhys            uint64
		AvailPhys            uint64
		TotalPageFile        uint64
		AvailPageFile        uint64
		TotalVirtual         uint64
		AvailVirtual         uint64
		AvailExtendedVirtual uint64
	}
	st.Length = uint32(unsafe.Sizeof(st))
	proc := syscall.NewLazyDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")
	r, _, _ := proc.Call(uintptr(unsafe.Pointer(&st)))
	if r == 0 {
		return 0
	}
	return int64(st.AvailPhys)
}
