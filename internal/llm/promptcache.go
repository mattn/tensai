package llm

// Prompt caching. An agent sends the same system prompt and the same tool
// definitions with every question -- thousands of tokens that prefill
// identically each time -- so the server keeps what it has already
// processed and starts from the first token that differs.
//
// Two things can be reused. A KV cache rolls back for free: its rows are
// written once and never revisited, so keeping a shorter slice of them is
// a valid state. A delta layer's recurrent state cannot roll back at all
// -- nothing in it is indexed by position -- so it is copied at the point
// two prompts diverge, and that copy is what a later prompt sharing the
// same opening restarts from.

// promptCache is what the server remembers between requests.
type promptCache struct {
	live []int // tokens the model's state currently holds
	// ckpt is a shorter prefix worth returning to: the tokens, and for a
	// model with recurrent layers the state at that point. The KV rows
	// need no copy, since nothing rewrites a row once it is written.
	ckpt      []int
	ckptDelta [][]float32
	hasDelta  bool
	// enabled is false where the state the server would roll back is not
	// the one it can reach: the GPU keeps its own resident cache.
	enabled bool
}

// commonPrefix is how many leading tokens two sequences share.
func commonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// plan decides where a prompt can start from. It returns the number of
// leading tokens already in the model's state, and whether the caller
// must first restore the checkpoint. A zero start means prefill it all.
func (c *promptCache) plan(tokens []int) (start int, restore bool) {
	if !c.enabled {
		return 0, false
	}
	// Extending what is already there needs nothing rolled back, which is
	// the one case a recurrent state can serve as well as a KV cache.
	if n := commonPrefix(c.live, tokens); n > 0 && n == len(c.live) && n < len(tokens) {
		return n, false
	} else if !c.hasDelta && n > 0 && n < len(tokens) {
		// Rolling a KV cache back is just a shorter slice of it.
		return n, false
	}
	if n := commonPrefix(c.ckpt, tokens); n > 0 && n == len(c.ckpt) && n < len(tokens) {
		return n, true
	}
	return 0, false
}

// snapshotDelta copies the recurrent state of every delta layer.
func snapshotDelta(m *qwen) [][]float32 {
	var out [][]float32
	for i := range m.blocks {
		b := &m.blocks[i]
		if b.delta == nil || b.dstate == nil {
			out = append(out, nil)
			continue
		}
		s := make([]float32, len(b.dstate.s)+len(b.dstate.conv))
		copy(s, b.dstate.s)
		copy(s[len(b.dstate.s):], b.dstate.conv)
		out = append(out, s)
	}
	return out
}

// restoreDelta puts a snapshot back. The KV cache is rolled back by the
// caller, which is a slice operation and needs no copy.
func restoreDelta(m *qwen, snap [][]float32) {
	for i := range m.blocks {
		b := &m.blocks[i]
		if b.delta == nil || i >= len(snap) || snap[i] == nil {
			continue
		}
		if b.dstate == nil {
			b.dstate = b.delta.newState()
		}
		n := len(b.dstate.s)
		copy(b.dstate.s, snap[i][:n])
		copy(b.dstate.conv, snap[i][n:])
	}
}
