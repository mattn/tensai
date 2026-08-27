//go:build darwin

package mmapfile

import (
	"syscall"
)

// Release tells the kernel the mapped range will not be needed again;
// darwin's syscall package lacks the Madvise wrapper, so this issues
// the raw system call with the same page-inward rounding as linux.
func Release(b []byte) {
	if len(b) == 0 {
		return
	}
	page := syscall.Getpagesize()
	start := 0
	if off := int(uintptr(addrOf(b)) % uintptr(page)); off != 0 {
		start = page - off
	}
	end := start + (len(b)-start)/page*page
	if end <= start {
		return
	}
	r := b[start:end]
	syscall.Syscall(syscall.SYS_MADVISE, uintptr(addrOf(r)), uintptr(len(r)), syscall.MADV_DONTNEED)
}
