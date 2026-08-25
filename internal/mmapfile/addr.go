package mmapfile

import "unsafe"

// addrOf returns the address of the first byte of b.
func addrOf(b []byte) unsafe.Pointer {
	return unsafe.Pointer(unsafe.SliceData(b))
}
