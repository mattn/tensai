//go:build linux || darwin

package mmapfile

import "syscall"

// Release tells the kernel the mapped range will not be needed again:
// its clean page-cache pages can drop immediately instead of crowding
// out the caller's own allocations. A later touch simply refaults from
// the file. b must lie within a mapping returned by Map; the range is
// rounded inward to page boundaries.
func Release(b []byte) {
	if len(b) == 0 {
		return
	}
	page := syscall.Getpagesize()
	// Round the start up and the end down so we never advise bytes
	// outside b (madvise needs page-aligned addresses).
	start := 0
	if off := int(uintptr(addrOf(b)) % uintptr(page)); off != 0 {
		start = page - off
	}
	end := start + (len(b)-start)/page*page
	if end <= start {
		return
	}
	syscall.Madvise(b[start:end], syscall.MADV_DONTNEED)
}
