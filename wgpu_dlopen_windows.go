//go:build wgpu && windows

package tensai

import "syscall"

// dlopenWGPU loads wgpu_native.dll. purego has no Dlopen on Windows — the
// handle LoadLibrary returns is what RegisterLibFunc feeds to
// GetProcAddress, so LoadLibrary is the whole port.
func dlopenWGPU(name string) (uintptr, error) {
	h, err := syscall.LoadLibrary(name)
	if err != nil {
		return 0, err
	}
	return uintptr(h), nil
}
