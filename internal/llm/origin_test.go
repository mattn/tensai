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
