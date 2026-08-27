package autograd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/mattn/tensai"
)

type paramsSnapshot struct {
	Params []*tensai.Matrix `json:"params"`
}

// SaveParams writes the values of autograd parameters as JSON, in the given
// order. Pass the same parameter list a Trainer uses, e.g.
// SaveParams(w, cell.Params()...).
func SaveParams(w io.Writer, params ...*Node) error {
	snap := paramsSnapshot{Params: make([]*tensai.Matrix, len(params))}
	for i, p := range params {
		if p == nil || p.Value == nil {
			return fmt.Errorf("tensai: save params: parameter %d is nil", i)
		}
		snap.Params[i] = p.Value
	}
	return json.NewEncoder(w).Encode(&snap)
}

// LoadParams restores parameter values saved by SaveParams. The parameters
// must be passed in the same order and have the same shapes as when saved.
func LoadParams(r io.Reader, params ...*Node) error {
	var snap paramsSnapshot
	if err := json.NewDecoder(r).Decode(&snap); err != nil {
		return fmt.Errorf("tensai: load params: %w", err)
	}
	if len(snap.Params) != len(params) {
		return fmt.Errorf("tensai: load params count mismatch: got %d, want %d",
			len(snap.Params), len(params))
	}
	for i, m := range snap.Params {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("tensai: load params %d: %w", i, err)
		}
		p := params[i]
		if p == nil || p.Value == nil {
			return fmt.Errorf("tensai: load params: parameter %d is nil", i)
		}
		if p.Value.Rows != m.Rows || p.Value.Cols != m.Cols {
			return fmt.Errorf("tensai: load params %d shape mismatch: got %dx%d, want %dx%d",
				i, m.Rows, m.Cols, p.Value.Rows, p.Value.Cols)
		}
		copy(p.Value.Data, m.Data)
	}
	return nil
}

// SaveParamsFile writes autograd parameters to a JSON file.
func SaveParamsFile(path string, params ...*Node) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := SaveParams(f, params...); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// LoadParamsFile restores autograd parameters from a file written by
// SaveParamsFile.
func LoadParamsFile(path string, params ...*Node) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return LoadParams(f, params...)
}
