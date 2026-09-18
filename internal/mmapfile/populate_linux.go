//go:build linux

package mmapfile

import "syscall"

// Populate asks the kernel to fault the whole mapping in now, in one
// pass, rather than a page at a time as the first sweep over the
// weights touches them: the pages are in the page cache already, and
// mapping them in bulk is several times cheaper than a fault apiece.
func Populate(b []byte) {
	if len(b) == 0 {
		return
	}
	syscall.Madvise(b, syscall.MADV_WILLNEED)
	// MADV_POPULATE_READ (5.14+) maps the pages; WILLNEED alone only
	// schedules the read. An older kernel returns EINVAL, which leaves
	// the demand faults as they were.
	const madvPopulateRead = 22
	syscall.Madvise(b, madvPopulateRead)
}
