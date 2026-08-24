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
	data := unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)
	closeFn := func() error {
		err := syscall.UnmapViewOfFile(addr)
		if cerr := syscall.CloseHandle(h); err == nil {
			err = cerr
		}
		return err
	}
	return data, closeFn, nil
}
