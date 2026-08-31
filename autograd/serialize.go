package autograd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/internal/dims"
)

// paramJSON is the on-disk form of one parameter. Two-dimensional
// parameters keep the Rows/Cols encoding checkpoints written before the
// engine went n-dimensional use, so those files still load (and still
// round-trip byte for byte); anything else carries its full shape.
type paramJSON struct {
	Rows  int            `json:"Rows,omitempty"`
	Cols  int            `json:"Cols,omitempty"`
	Shape []int          `json:"Shape,omitempty"`
	Data  []tensai.Float `json:"Data"`
}

// shape returns the parameter's shape in either encoding.
func (p *paramJSON) shape() []int {
	if len(p.Shape) > 0 {
		return p.Shape
	}
	return []int{p.Rows, p.Cols}
}

type paramsSnapshot struct {
	Params []paramJSON `json:"params"`
}

// SaveParams writes the values of autograd parameters as JSON, in the given
// order. Pass the same parameter list a Trainer uses, e.g.
// SaveParams(w, cell.Params()...).
func SaveParams(w io.Writer, params ...*Node) error {
	snap := paramsSnapshot{Params: make([]paramJSON, len(params))}
	for i, p := range params {
		if p == nil || p.Value == nil {
			return fmt.Errorf("tensai: save params: parameter %d is nil", i)
		}
		t := p.Value
		if len(t.Shape) == 2 {
			snap.Params[i] = paramJSON{Rows: t.Shape[0], Cols: t.Shape[1], Data: t.Data}
			continue
		}
		snap.Params[i] = paramJSON{Shape: t.Shape, Data: t.Data}
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
	for i := range snap.Params {
		saved := &snap.Params[i]
		shape := saved.shape()
		if len(saved.Data) != dims.Prod(shape) {
			return fmt.Errorf("tensai: load params %d: %s has %d elements",
				i, shapeString(shape), len(saved.Data))
		}
		p := params[i]
		if p == nil || p.Value == nil {
			return fmt.Errorf("tensai: load params: parameter %d is nil", i)
		}
		if !dims.Same(p.Value.Shape, shape) {
			return fmt.Errorf("tensai: load params %d shape mismatch: got %s, want %s",
				i, shapeString(shape), shapeString(p.Value.Shape))
		}
		copy(p.Value.Data, saved.Data)
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
