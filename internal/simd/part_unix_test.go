//go:build goexperiment.simd && amd64 && unix

package simd

import (
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// TestPartLoadAtPageEnd puts a short slice flush against an unmapped page
// and loads it: a part load that reads a whole vector and masks afterwards
// faults here, which is what the Windows runners hit on ordinary heap
// slices.
func TestPartLoadAtPageEnd(t *testing.T) {
	page := os.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mem)
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	for n := 1; n < 8; n++ {
		s := unsafe.Slice((*float32)(unsafe.Pointer(&mem[page-4*n])), n)
		for i := range s {
			s[i] = float32(i + 1)
		}
		var got [8]float32
		StoreF32x8(LoadF32x8Part(s), got[:])
		for i := range got {
			want := float32(0)
			if i < n {
				want = float32(i + 1)
			}
			if got[i] != want {
				t.Fatalf("n=%d lane %d: %v, want %v", n, i, got[i], want)
			}
		}
	}
}
