package tensai

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// stateful is implemented by layers with non-parameter state that must
// survive a save/load round trip (e.g. BatchNorm running statistics).
type stateful interface {
	extraState() map[string][]Float
	setExtraState(map[string][]Float) error
}

type layerSnapshot struct {
	Type    string             `json:"type"`
	Weights *Matrix            `json:"weights,omitempty"`
	Bias    []Float            `json:"bias,omitempty"`
	Extra   map[string][]Float `json:"extra,omitempty"`
}

type modelSnapshot struct {
	Layers []layerSnapshot `json:"layers"`
}

// Save writes the model's parameters as JSON. The architecture itself is not
// stored: Load must be called on a model built and compiled with the same
// layers.
func (s *Sequential) Save(w io.Writer) error {
	if s.optimizer == nil {
		return fmt.Errorf("tensai: save requires a compiled model")
	}
	snap := modelSnapshot{Layers: make([]layerSnapshot, len(s.layers))}
	for i, l := range s.layers {
		weights, bias := l.Params()
		snap.Layers[i] = layerSnapshot{Type: fmt.Sprintf("%T", l), Weights: weights, Bias: bias}
		if st, ok := l.(stateful); ok {
			snap.Layers[i].Extra = st.extraState()
		}
	}
	enc := json.NewEncoder(w)
	return enc.Encode(&snap)
}

// Load restores parameters saved by Save into a model compiled with the same
// architecture.
func (s *Sequential) Load(r io.Reader) error {
	if s.optimizer == nil {
		return fmt.Errorf("tensai: load requires a compiled model")
	}
	var snap modelSnapshot
	if err := json.NewDecoder(r).Decode(&snap); err != nil {
		return fmt.Errorf("tensai: load: %w", err)
	}
	if len(snap.Layers) != len(s.layers) {
		return fmt.Errorf("tensai: load layer count mismatch: model=%d snapshot=%d",
			len(s.layers), len(snap.Layers))
	}
	for i, l := range s.layers {
		ls := snap.Layers[i]
		if got := fmt.Sprintf("%T", l); got != ls.Type {
			return fmt.Errorf("tensai: load layer %d type mismatch: model=%s snapshot=%s", i, got, ls.Type)
		}
		if ls.Weights != nil {
			if err := ls.Weights.Validate(); err != nil {
				return fmt.Errorf("tensai: load layer %d: %w", i, err)
			}
			if err := l.SetParams(ls.Weights, ls.Bias); err != nil {
				return fmt.Errorf("tensai: load layer %d: %w", i, err)
			}
		}
		if st, ok := l.(stateful); ok {
			if err := st.setExtraState(ls.Extra); err != nil {
				return fmt.Errorf("tensai: load layer %d: %w", i, err)
			}
		}
	}
	return nil
}

// SaveFile writes the model's parameters to a JSON file.
func (s *Sequential) SaveFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := s.Save(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// LoadFile restores parameters from a file written by SaveFile.
func (s *Sequential) LoadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return s.Load(f)
}

type paramsSnapshot struct {
	Params []*Matrix `json:"params"`
}

// SaveParams writes the values of autograd parameters as JSON, in the given
// order. Pass the same parameter list a Trainer uses, e.g.
// SaveParams(w, cell.Params()...).
func SaveParams(w io.Writer, params ...*Node) error {
	snap := paramsSnapshot{Params: make([]*Matrix, len(params))}
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
