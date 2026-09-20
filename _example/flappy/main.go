// Flappy Bird played by a language model, to put a number on a question:
// what can a model decide at a reflex game when each step is one scored
// question? Nothing is trained. The model reads the state as a sentence
// and answers through Engine.Score, so the answer is a probability and
// the cost is one prefill per decision; a random flapper and a one-line
// heuristic play the same game as the floor and the ceiling. Asked yes
// or no, no model plays; asked which of two numbers is larger, with the
// numbers as the options, a 1B plays the heuristic's game.
//
//	go run ./_example/flappy                    # Qwen2.5-0.5B-Instruct from the cache
//	go run ./_example/flappy -model ./x.gguf    # another model
//	go run ./_example/flappy -larger -rows      # the comparison asked as a choice of numbers
//	go run ./_example/flappy -episodes 5 -show  # print each decision
//	go run ./_example/flappy -larger -nobase -screen   # watch it in the terminal
package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
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
	g := &game{y: 50, rng: rand.New(rand.NewPCG(uint64(seed), 0)), nextX: 40}
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

// rowHeuristic is the heuristic on heights rounded to one digit, the
// most the -rows variant could do if the model compared perfectly.
type rowHeuristic struct{}

func (rowHeuristic) name() string { return "heur/rows" }
func (rowHeuristic) flap(g *game) (bool, string) {
	p := g.next()
	return math.Round(g.y/10) < math.Round((p.gapLow+pipeGap/2-2)/10), ""
}

// model asks the language model, every step, and flaps when yes wins.
type model struct {
	e         *llm.Engine
	threshold float64
	hint      bool
	compare   bool
	// larger asks the comparison with the two numbers as the options:
	// "which is larger, 57 or 63?", scored over "57" and "63". The
	// answer is then a number the model writes, not a yes it leans to.
	larger bool
	// rows quantizes heights to one digit each before asking.
	rows bool
}

func (m model) name() string {
	switch {
	case m.larger && m.rows:
		return "model+row"
	case m.larger:
		return "model+lgr"
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
	if m.larger {
		return m.flapLarger(g)
	}
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

// flapLarger asks which of two numbers is larger, the bird's height and
// the middle of the opening, and flaps when the model says the middle.
// The heuristic's own margin is kept: the target sits 2 under the middle.
func (m model) flapLarger(g *game) (bool, string) {
	p := g.next()
	bird, mid := g.y, p.gapLow+pipeGap/2-2
	if m.rows {
		bird, mid = math.Round(bird/10), math.Round(mid/10)
	}
	a, b := fmt.Sprintf("%.0f", bird), fmt.Sprintf("%.0f", mid)
	if a == b {
		return false, "same"
	}
	// Asked both ways round, so the order the numbers come in, which a
	// small model leans on when they are close, cancels out.
	var pm float64
	for _, pair := range [][2]string{{a, b}, {b, a}} {
		q := fmt.Sprintf("Which number is larger, %s or %s? Answer with the number only.", pair[0], pair[1])
		probs, err := m.e.Score(q, []string{a, b})
		if err != nil {
			return false, err.Error()
		}
		pm += probs[1] / 2
	}
	return pm > m.threshold, fmt.Sprintf("%s>%s %.0f%%", b, a, 100*pm)
}

// The screen: the world drawn in text, redrawn in place with escape
// sequences, so a game can be watched. Each row is rows units of
// height and each column cols units of distance; the bird sits at
// column birdCol.
const (
	screenRows = 25
	screenCols = 64
	rowUnits   = height / screenRows
	colUnits   = scroll
	birdCol    = 6
)

// draw paints the current frame over the previous one. The first frame
// clears the terminal and hides the cursor; the caller shows it again.
func (g *game) draw(who string, flap bool, note string) {
	var sb strings.Builder
	sb.WriteString("\x1b[H")
	sb.WriteString(fmt.Sprintf("\x1b[2K %-12s pipes %-4d step %-4d %s\n", who, g.score, g.steps, note))
	birdRow := int((height - g.y) / rowUnits)
	for r := 0; r < screenRows; r++ {
		sb.WriteString("\x1b[2K")
		lo, hi := height-float64(r+1)*rowUnits, height-float64(r)*rowUnits // heights this row spans
		for c := 0; c < screenCols; c++ {
			x := float64(c-birdCol) * colUnits
			ch := ' '
			for _, p := range g.pipes {
				if x >= p.x-birdSize && x <= p.x+birdSize {
					if hi <= p.gapLow || lo >= p.gapLow+pipeGap {
						ch = '#'
					}
					break
				}
			}
			if c == birdCol && r == birdRow {
				ch = '@'
				if flap {
					ch = '^'
				}
				if g.dead {
					ch = 'x'
				}
			}
			sb.WriteRune(ch)
		}
		sb.WriteByte('\n')
	}
	sb.WriteString(strings.Repeat("=", screenCols) + "\n")
	os.Stdout.WriteString(sb.String())
}

func play(p player, seed int64, limit int, show, screen bool) (score, steps int) {
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
		if screen {
			g.draw(p.name(), flap, note)
			// A player that answers at once is slowed to a watchable
			// pace; a model sets its own.
			time.Sleep(40 * time.Millisecond)
		}
		g.step(flap)
	}
	if screen {
		g.draw(p.name(), false, map[bool]string{true: "dead", false: "win"}[g.dead])
		time.Sleep(1500 * time.Millisecond)
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
	screen := flag.Bool("screen", false, "draw the game in the terminal as it is played")
	skipModel := flag.Bool("nomodel", false, "run the baselines only")
	skipBase := flag.Bool("nobase", false, "skip the baselines and play the model only")
	hint := flag.Bool("hint", false, "also play the model with the relative-position hint in its state")
	compare := flag.Bool("compare", false, "also play the model asked only the heuristic's comparison")
	larger := flag.Bool("larger", false, "also play the model asked which of two numbers is larger, the numbers being the options")
	rows := flag.Bool("rows", false, "also play the -larger variant with heights rounded to one digit")
	flag.Parse()

	var players []player
	if !*skipBase {
		players = []player{randomPlayer{rng: rand.New(rand.NewPCG(1, 0))}, heuristic{}, rowHeuristic{}}
	}
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
		if *larger {
			players = append(players, model{e: e, threshold: *threshold, larger: true})
		}
		if *rows {
			players = append(players, model{e: e, threshold: *threshold, larger: true, rows: true})
		}
	}

	// With the screen on, the table waits until the games are over and
	// goes under the last frame; otherwise it grows a row per player.
	table := fmt.Sprintf("%-10s %8s %8s %10s\n", "player", "pipes", "steps", "per step")
	if *screen {
		os.Stdout.WriteString("\x1b[2J\x1b[?25l")
		defer func() {
			fmt.Printf("\x1b[%d;1H\x1b[?25h%s", screenRows+3, table)
		}()
	} else {
		os.Stdout.WriteString(table)
		table = ""
	}
	for _, p := range players {
		var pipes, steps int
		start := time.Now()
		for ep := 0; ep < *episodes; ep++ {
			if *show {
				fmt.Printf("%s, episode %d\n", p.name(), ep+1)
			}
			s, n := play(p, int64(100+ep), *limit, *show, *screen)
			pipes += s
			steps += n
		}
		per := time.Duration(0)
		if steps > 0 {
			per = time.Since(start) / time.Duration(steps)
		}
		row := fmt.Sprintf("%-10s %8.1f %8.1f %10v\n", p.name(), float64(pipes)/float64(*episodes), float64(steps)/float64(*episodes), per.Round(time.Millisecond))
		if *screen {
			table += row
		} else {
			os.Stdout.WriteString(row)
		}
	}
}
