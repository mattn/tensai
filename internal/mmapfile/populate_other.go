//go:build !linux

package mmapfile

// Populate is a no-op where the platform offers no bulk fault-in.
func Populate(b []byte) {}
