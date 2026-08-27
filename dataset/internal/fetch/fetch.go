// Package fetch holds the download-and-cache plumbing shared by the
// built-in dataset loaders.
package fetch

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// DefaultDir returns the cache directory for the named dataset,
// os.UserCacheDir()/tensai/<name>.
func DefaultDir(name string) (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("tensai: %s: %w", name, err)
	}
	return filepath.Join(cache, "tensai", name), nil
}

// Download fetches url into path atomically (via a temp file and rename),
// announcing the transfer on stderr.
func Download(url, path string) error {
	fmt.Fprintf(os.Stderr, "tensai: downloading %s\n", url)
	client := http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("tensai: %s: %s", url, resp.Status)
	}

	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, path)
}
