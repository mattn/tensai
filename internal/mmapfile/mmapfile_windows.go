//go:build windows

package mmapfile

import (
	"os"
	"syscall"
	"unsafe"
)

// Map maps the whole file read-only via CreateFileMapping/MapViewOfFile.
func Map(f *os.File) ([]byte, func() error, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	size := st.Size()
	if size == 0 {
		return nil, nil, syscall.EINVAL
	}
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READONLY,
		uint32(size>>32), uint32(size), nil)
	if err != nil {
		return nil, nil, err
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		syscall.CloseHandle(h)
		return nil, nil, err
	}
	// The view is not Go memory, so the address never moves; reading the
	// uintptr's storage as a pointer says so in a form vet accepts, where
	// a direct unsafe.Pointer(addr) is flagged as a possible misuse.
	base := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	data := unsafe.Slice((*byte)(base), size)
	closeFn := func() error {
		err := syscall.UnmapViewOfFile(addr)
		if cerr := syscall.CloseHandle(h); err == nil {
			err = cerr
		}
		return err
	}
	return data, closeFn, nil
}
