//go:build !linux && !darwin && !windows

package mmapfile

import (
	"errors"
	"os"
)

// Map is unsupported on this platform; callers fall back to ReadAt.
func Map(f *os.File) ([]byte, func() error, error) {
	return nil, nil, errors.New("mmap unsupported")
}
