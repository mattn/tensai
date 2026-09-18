//go:build !linux && !darwin && !windows

package sysmem

// Available is unknown here.
func Available() int64 { return 0 }
