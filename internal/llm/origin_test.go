package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOriginRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if got := Origin(dir); got != "" {
		t.Errorf("a directory with no record answered %q", got)
	}
	recordOrigin(dir, "Qwen/Qwen3-4B-Instruct-2507")
	if got := Origin(dir); got != "Qwen/Qwen3-4B-Instruct-2507" {
		t.Errorf("Origin = %q after recording it", got)
	}
	// Writing the same origin again must not disturb the file, which the
	// listing reads as the model's own modification time.
	recordOrigin(dir, "Qwen/Qwen3-4B-Instruct-2507")
	if got := Origin(dir); got != "Qwen/Qwen3-4B-Instruct-2507" {
		t.Errorf("Origin = %q after a second record", got)
	}
}

// A bare name is not a repo: it cannot be handed to -model on a machine
// that does not have the model yet, which is the whole point of keeping it.
func TestOriginRejectsWhatCannotBeFetched(t *testing.T) {
	for _, bad := range []string{
		"Qwen3-4B",   // no organization
		"a/b/c",      // not a repo path
		"../escape",  // traversal
		`Qwen\Qwen3`, // a Windows path, not a repo
		"/absolute",  // empty organization
		"",           // nothing at all
	} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, originFile), []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := Origin(dir); got != "" {
			t.Errorf("Origin(%q) = %q, want it refused", bad, got)
		}
	}
}

// Nothing is recorded for a model that was never downloaded from a repo.
func TestRecordOriginNeedsARepo(t *testing.T) {
	dir := t.TempDir()
	recordOrigin(dir, "Qwen3-4B-Instruct-2507")
	if _, err := os.Stat(filepath.Join(dir, originFile)); !os.IsNotExist(err) {
		t.Errorf("a bare name was recorded as an origin")
	}
}

func TestGGUFOrigin(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(p, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := GGUFOrigin(p); got != "" {
		t.Fatalf("no sidecar: got %q", got)
	}
	recordGGUFOrigin(p, "org/repo")
	if got := GGUFOrigin(p); got != "org/repo" {
		t.Fatalf("got %q, want org/repo", got)
	}
	for _, bad := range []string{"norepo", "a/b/c", "..", `or\g/repo`, "org/.."} {
		if err := os.WriteFile(p+originSidecar, []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if got := GGUFOrigin(p); got != "" {
			t.Fatalf("sidecar %q: got %q, want empty", bad, got)
		}
	}
}

// Two organizations can publish the same name; each gets its own
// directory, and a download from before the org joined the path is kept
// only by the repo it records.
func TestDefaultDataDirKeepsTheOrg(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())         // macOS
	t.Setenv("LocalAppData", t.TempDir()) // Windows
	root := CacheRoot()
	if got, want := DefaultDataDir("Qwen/Qwen3-4B"), filepath.Join(root, "Qwen", "Qwen3-4B"); got != want {
		t.Errorf("fresh download: %s, want %s", got, want)
	}
	legacy := filepath.Join(root, "Qwen3-4B")
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	recordOrigin(legacy, "Qwen/Qwen3-4B")
	if got := DefaultDataDir("Qwen/Qwen3-4B"); got != legacy {
		t.Errorf("recorded legacy download: %s, want %s", got, legacy)
	}
	if got, want := DefaultDataDir("other/Qwen3-4B"), filepath.Join(root, "other", "Qwen3-4B"); got != want {
		t.Errorf("same name from another org: %s, want %s", got, want)
	}
	if got := DefaultDataDir("a/../../escape"); filepath.Dir(got) != root {
		t.Errorf("a path that is no repo left the cache root: %s", got)
	}
}
