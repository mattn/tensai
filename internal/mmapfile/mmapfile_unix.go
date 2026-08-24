//go:build linux || darwin

// Package mmapfile memory-maps files read-only, so checkpoint readers can
// slice tensor bytes straight out of the page cache instead of copying
// them through read buffers.
package mmapfile

import (
	"os"
	"syscall"
)

// Map maps the whole file read-only. The returned close function unmaps;
// the byte slice must not be used after it. Returns an error on empty
// files or platforms without support, in which case callers fall back to
// ReadAt.
func Map(f *os.File) ([]byte, func() error, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if st.Size() == 0 {
		return nil, nil, syscall.EINVAL
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
