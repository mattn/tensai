//go:build (wgpu || wgpu24) && (linux || darwin)

package gpu

import "github.com/ebitengine/purego"

// dlopenWGPU loads the wgpu-native shared library. RTLD_GLOBAL keeps its
// symbols visible to anything the driver dlopens behind our back.
func dlopenWGPU(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
