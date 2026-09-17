// Flappy Bird without a screen, played three ways, to put a number on a
// question: how far does a language model get at a reflex game when it
// is only asked, each step, whether to flap? Nothing is trained. The
// model reads the state as a sentence and answers yes or no through
// Engine.Score, so the answer is a probability and the cost is one
// prefill per decision; a random flapper and a one-line heuristic play
// the same game as the floor and the ceiling.
//
//	go run ./_example/flappy                    # Qwen2.5-0.5B-Instruct from the cache
//	go run ./_example/flappy -model ./x.gguf    # another model
//	go run ./_example/flappy -episodes 5 -show  # print each decision
package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mattn/tensai/internal/llm"
)

// The world: heights run 0 (ground) to 100 (ceiling), the bird sits at
// x = 0 and pipes scroll toward it. Units are arbitrary; the constants
// are tuned so that a bird which never flaps dies in about a second and
// one that flaps every step hits the ceiling nearly as fast, which is the
// shape of the real game.
const (
	height   = 100.0
	gravity  = 1.2  // per step, downward
	flapV    = -7.0 // vertical speed a flap sets
	scroll   = 3.0  // how far pipes move toward the bird per step
	pipeGap  = 30.0 // height of the opening
	pipeEver = 60.0 // distance between pipes
	birdSize = 4.0
)

type pipe struct {
	x, gapLow float64 // gap spans [gapLow, gapLow+pipeGap]
}

type game struct {
	y, vy   float64
	pipes   []pipe
	score   int
	steps   int
	rng     *rand.Rand
	nextX   float64
	dead    bool
	lastMsg string
}

func newGame(seed int64) *game {
	g := &game{y: 50, rng: rand.New(rand.NewSource(seed)), nextX: 40}
	for i := 0; i < 3; i++ {
		g.spawn()
	}
	return g
}

func (g *game) spawn() {
	g.pipes = append(g.pipes, pipe{x: g.nextX, gapLow: 15 + g.rng.Float64()*(height-30-pipeGap)})
	g.nextX += pipeEver
}

// next is the first pipe the bird has not passed.
func (g *game) next() pipe {
	for _, p := range g.pipes {
		if p.x+birdSize >= 0 {
			return p
		}
	}
	return g.pipes[len(g.pipes)-1]
}

func (g *game) step(flap bool) {
	if flap {
		g.vy = flapV
	}
	g.vy += gravity
	g.y -= g.vy
	for i := range g.pipes {
		g.pipes[i].x -= scroll
	}
	g.steps++
	if g.y <= 0 || g.y >= height {
		g.dead = true
		return
	}
	for _, p := range g.pipes {
		if p.x-birdSize <= 0 && p.x+birdSize >= 0 {
			if g.y < p.gapLow || g.y > p.gapLow+pipeGap {
				g.dead = true
				return
			}
		}
	}
	if g.pipes[0].x+birdSize < -scroll {
		g.pipes = g.pipes[1:]
		g.score++
		g.spawn()
	}
	g.nextX -= scroll
}

// state is what a player sees, in words. With hint, one more sentence
// says where the bird is relative to the middle of the opening, which
// is the whole comparison the heuristic makes: a model given it is being
// handed the decision, and how much that helps says how much of the
// judgment was the model's.
func (g *game) state(hint bool) string {
	p := g.next()
	dir := "falling"
	if g.vy < 0 {
		dir = "rising"
	}
	s := fmt.Sprintf("The bird is at height %.0f on a scale where 0 is the ground and 100 is the ceiling, and it is %s at %.0f per step. "+
		"The next pipe is %.0f ahead; its opening spans heights %.0f to %.0f, and the bird must pass through it. "+
		"Flapping makes the bird rise sharply for a moment, then it falls again.",
		g.y, dir, absf(g.vy), p.x, p.gapLow, p.gapLow+pipeGap)
	if hint {
		mid := p.gapLow + pipeGap/2
		rel := "below"
		if g.y > mid {
			rel = "above"
		}
		s += fmt.Sprintf(" Right now the bird is %.0f %s the middle of the opening.", absf(g.y-mid), rel)
	}
	return s
}

func absf(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

type player interface {
	name() string
	flap(g *game) (bool, string)
}

type randomPlayer struct{ rng *rand.Rand }

func (randomPlayer) name() string { return "random" }
func (r randomPlayer) flap(*game) (bool, string) {
	return r.rng.Float64() < 0.15, ""
}

// heuristic is the whole intelligence of the game in one comparison:
// flap when below the middle of the opening.
type heuristic struct{}

func (heuristic) name() string { return "heuristic" }
func (heuristic) flap(g *game) (bool, string) {
	p := g.next()
	return g.y < p.gapLow+pipeGap/2-2, ""
}

// model asks the language model, every step, and flaps when yes wins.
type model struct {
	e         *llm.Engine
	threshold float64
	hint      bool
	compare   bool
}

func (m model) name() string {
	switch {
	case m.compare:
		return "model+cmp"
	case m.hint:
		return "model+hint"
	}
	return "model"
}

// flap asks the model. The plain question is the action itself; with
// compare it is the heuristic's comparison in words, "is the bird below
// the middle of the opening", and the code turns yes into a flap: that
// leaves the model only a numeric comparison to get right, which is the
// least it could be asked.
func (m model) flap(g *game) (bool, string) {
	q := g.state(m.hint) + " Should the bird flap right now? Answer yes or no."
	if m.compare {
		p := g.next()
		q = fmt.Sprintf("The bird is at height %.0f. The opening in the next pipe spans heights %.0f to %.0f, so its middle is at %.0f. Is the bird below the middle of the opening? Answer yes or no.",
			g.y, p.gapLow, p.gapLow+pipeGap, p.gapLow+pipeGap/2)
	}
	probs, err := m.e.Score(q, []string{"yes", "no"})
	if err != nil {
		return false, err.Error()
	}
	return probs[0] > m.threshold, fmt.Sprintf("yes %.0f%%", 100*probs[0])
}

func play(p player, seed int64, limit int, show bool) (score, steps int) {
	g := newGame(seed)
	for !g.dead && g.steps < limit {
		flap, note := p.flap(g)
		if show {
			mark := " "
			if flap {
				mark = "^"
			}
			fmt.Printf("  %s y=%5.1f vy=%5.1f pipe=%4.0f gap=%3.0f-%3.0f %s\n", mark, g.y, g.vy, g.next().x, g.next().gapLow, g.next().gapLow+pipeGap, note)
		}
		g.step(flap)
	}
	return g.score, g.steps
}

func main() {
	modelPath := flag.String("model", filepath.Join(llm.CacheRoot(), "Qwen2.5-0.5B-Instruct"), "model directory or .gguf")
	bits := flag.Int("bits", 8, "weight bits: 8 or 4")
	episodes := flag.Int("episodes", 3, "games per player")
	limit := flag.Int("limit", 400, "steps per game before it is called a win")
	threshold := flag.Float64("threshold", 0.5, "flap when P(yes) exceeds this")
	show := flag.Bool("show", false, "print every decision")
	skipModel := flag.Bool("nomodel", false, "run the two baselines only")
	hint := flag.Bool("hint", false, "also play the model with the relative-position hint in its state")
	compare := flag.Bool("compare", false, "also play the model asked only the heuristic's comparison")
	flag.Parse()

	players := []player{randomPlayer{rng: rand.New(rand.NewSource(1))}, heuristic{}}
	if !*skipModel {
		opts := llm.Options{Bits: *bits, Log: io.Discard}
		if strings.HasSuffix(*modelPath, ".gguf") {
			opts.GGUF = *modelPath
		} else {
			opts.Data = *modelPath
		}
		e, err := llm.Open(opts)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer e.Close()
		players = append(players, model{e: e, threshold: *threshold})
		if *hint {
			players = append(players, model{e: e, threshold: *threshold, hint: true})
		}
		if *compare {
			players = append(players, model{e: e, threshold: *threshold, compare: true})
		}
	}

	fmt.Printf("%-10s %8s %8s %10s\n", "player", "pipes", "steps", "per step")
	for _, p := range players {
		var pipes, steps int
		start := time.Now()
		for ep := 0; ep < *episodes; ep++ {
			if *show {
				fmt.Printf("%s, episode %d\n", p.name(), ep+1)
			}
			s, n := play(p, int64(100+ep), *limit, *show)
			pipes += s
			steps += n
		}
		per := time.Duration(0)
		if steps > 0 {
			per = time.Since(start) / time.Duration(steps)
		}
		fmt.Printf("%-10s %8.1f %8.1f %10v\n", p.name(), float64(pipes)/float64(*episodes), float64(steps)/float64(*episodes), per.Round(time.Millisecond))
	}
}
