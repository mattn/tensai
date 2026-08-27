package model

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/mattn/tensai"
	"github.com/mattn/tensai/layer"
)

// stateful is implemented by layers with non-parameter state that must
// survive a save/load round trip (e.g. BatchNorm running statistics).
type stateful interface {
	ExtraState() map[string][]tensai.Float
	SetExtraState(map[string][]tensai.Float) error
}

// layerTypeName is the identifier stored in snapshots for a layer: the bare
// type name with any pointer marker and package path stripped, so files
// written before the package split ("*tensai.Dense") still match today's
// "*layer.Dense".
func layerTypeName(l layer.Layer) string {
	name := fmt.Sprintf("%T", l)
	if i := strings.LastIndexByte(name, '.'); i >= 0 {
		name = name[i+1:]
	}
	return name
}

// trimTypeName normalizes a stored snapshot type the same way.
func trimTypeName(name string) string {
	if i := strings.LastIndexByte(name, '.'); i >= 0 {
		name = name[i+1:]
	}
	return name
}

type layerSnapshot struct {
	Type    string                    `json:"type"`
	Weights *tensai.Matrix            `json:"weights,omitempty"`
	Bias    []tensai.Float            `json:"bias,omitempty"`
	Extra   map[string][]tensai.Float `json:"extra,omitempty"`
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
		snap.Layers[i] = layerSnapshot{Type: layerTypeName(l), Weights: weights, Bias: bias}
		if st, ok := l.(stateful); ok {
			snap.Layers[i].Extra = st.ExtraState()
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
		if got := layerTypeName(l); got != trimTypeName(ls.Type) {
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
			if err := st.SetExtraState(ls.Extra); err != nil {
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
