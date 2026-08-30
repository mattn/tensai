//go:build !amd64

package workpool

import "runtime"

func pause() { runtime.Gosched() }
