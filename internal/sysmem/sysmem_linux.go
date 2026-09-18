//go:build linux

package sysmem

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// Available is the kernel's MemAvailable: what could be given to a new
// allocation without swapping, page cache it is willing to drop
// included. Zero when /proc/meminfo cannot be read.
func Available() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}
