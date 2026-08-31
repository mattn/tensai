package llm

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// body is the file under test: big enough that a half-served attempt
// leaves something worth resuming.
var body = []byte(strings.Repeat("tensai/", 4096))

// rangeServer serves body with ETag and Range support, and hands each
// request to fail(n) first so a test can decide how attempt n misbehaves.
// Returning true means the handler already dealt with the request.
func rangeServer(etag func() string, fail func(n int64, w http.ResponseWriter) bool) *httptest.Server {
	var calls int64
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt64(&calls, 1)
		tag := etag()
		off := 0
		if rng := r.Header.Get("Range"); rng != "" {
			// A stale If-Range means the client's partial is not a prefix
			// of what we would send, so answer with the whole file.
			if want := r.Header.Get("If-Range"); want == "" || want == tag {
				fmt.Sscanf(rng, "bytes=%d-", &off)
			}
		}
		if off > len(body) {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		w.Header().Set("ETag", tag)
		w.Header().Set("Content-Length", strconv.Itoa(len(body)-off))
		if off > 0 {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", off, len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
		}
		if fail != nil && fail(n, w) {
			return
		}
		w.Write(body[off:])
	}))
}

func fetched(t *testing.T, srv *httptest.Server) (string, error) {
	t.Helper()
	dir := t.TempDir()
	p, err := fetch(srv.URL+"/", dir, "model.bin")
	if err != nil {
		return dir, err
	}
	got, rerr := os.ReadFile(p)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if string(got) != string(body) {
		t.Fatalf("downloaded %d bytes, want %d", len(got), len(body))
	}
	// Nothing is left behind once the file is whole.
	for _, leftover := range []string{"model.bin.tmp", "model.bin.tmp.etag"} {
		if _, err := os.Stat(filepath.Join(dir, leftover)); !os.IsNotExist(err) {
			t.Errorf("%s survived a finished download", leftover)
		}
	}
	return dir, nil
}

func TestFetchPlain(t *testing.T) {
	srv := rangeServer(func() string { return `"v1"` }, nil)
	defer srv.Close()
	if _, err := fetched(t, srv); err != nil {
		t.Fatal(err)
	}
}

// A connection that dies mid-body must be picked up where it stopped,
// not started over: the point of the exercise on a multi-gigabyte file.
func TestFetchResumesAfterATruncatedBody(t *testing.T) {
	var resumedAt int64
	srv := rangeServer(func() string { return `"v1"` }, func(n int64, w http.ResponseWriter) bool {
		if n == 1 {
			w.Write(body[:len(body)/2])
			// Abort without the rest, as a dropped connection would.
			panic(http.ErrAbortHandler)
		}
		return false
	})
	defer srv.Close()
	// An aborted handler logs; the abort is the point of the test.
	srv.Config.ErrorLog = log.New(io.Discard, "", 0)
	// The second request must ask for the tail.
	srv.Config.Handler = logRange(srv.Config.Handler, &resumedAt)
	if _, err := fetched(t, srv); err != nil {
		t.Fatal(err)
	}
	if resumedAt == 0 {
		t.Error("the retry re-downloaded from the start instead of resuming")
	}
}

func logRange(h http.Handler, at *int64) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rng := r.Header.Get("Range"); rng != "" {
			var off int64
			fmt.Sscanf(rng, "bytes=%d-", &off)
			if off > 0 {
				atomic.StoreInt64(at, off)
			}
		}
		h.ServeHTTP(w, r)
	})
}

// A file that changed upstream must not be spliced onto the old prefix.
func TestFetchRestartsWhenTheFileChanged(t *testing.T) {
	var calls int64
	srv := rangeServer(
		func() string {
			if atomic.LoadInt64(&calls) == 0 {
				return `"v1"`
			}
			return `"v2"`
		},
		func(n int64, w http.ResponseWriter) bool {
			if n == 1 {
				atomic.StoreInt64(&calls, 1)
				w.Write(body[:len(body)/2])
				panic(http.ErrAbortHandler)
			}
			return false
		})
	defer srv.Close()
	if _, err := fetched(t, srv); err != nil {
		t.Fatal(err)
	}
}

// A server that is briefly unhappy gets another chance; one that says the
// file is not there does not.
func TestFetchRetriesServerErrors(t *testing.T) {
	srv := rangeServer(func() string { return `"v1"` }, func(n int64, w http.ResponseWriter) bool {
		if n <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return true
		}
		return false
	})
	defer srv.Close()
	if _, err := fetched(t, srv); err != nil {
		t.Fatal(err)
	}
}

func TestFetchDoesNotRetryNotFound(t *testing.T) {
	var calls int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	if _, err := fetch(srv.URL+"/", t.TempDir(), "model.bin"); err == nil {
		t.Fatal("a 404 was treated as a download")
	}
	if n := atomic.LoadInt64(&calls); n != 1 {
		t.Errorf("a 404 was retried %d times", n-1)
	}
}

func TestFetchGGUFRejectsBadRefs(t *testing.T) {
	for _, ref := range []string{
		"file.gguf",              // no repo
		"repo/file.gguf",         // no org
		"org/repo/sub/file.gguf", // nested path
		"org/repo/file",          // not a gguf
		"org//file.gguf",         // empty repo
		"//file.gguf",            // empty org and repo
		"org/repo/.gguf",         // no file name
	} {
		if _, err := FetchGGUF(ref); err == nil {
			t.Errorf("FetchGGUF(%q): expected an error", ref)
		}
	}
}
