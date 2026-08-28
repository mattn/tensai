package llm

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Without a home directory in the environment, os.UserCacheDir gives up;
// the account still knows where home is, and a multi-gigabyte download
// belongs there rather than in the working directory.
func TestCacheRootWithoutHomeEnv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("UserCacheDir reads %LocalAppData% on windows")
	}
	t.Setenv("HOME", "")
	t.Setenv("XDG_CACHE_HOME", "")
	got := CacheRoot()
	if !filepath.IsAbs(got) {
		t.Fatalf("CacheRoot() = %q, want an absolute path, not one relative to the working directory", got)
	}
	if !strings.HasSuffix(got, filepath.Join(".cache", "tensai")) {
		t.Fatalf("CacheRoot() = %q, want it under the account's .cache", got)
	}
	if _, err := os.Stat(filepath.Dir(filepath.Dir(got))); err != nil {
		t.Fatalf("CacheRoot() = %q, whose home directory does not exist: %v", got, err)
	}
}
