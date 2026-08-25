//go:build !linux && !darwin

package mmapfile

// Release is a no-op where madvise is unavailable.
func Release(b []byte) {}
