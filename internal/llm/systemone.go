package llm

// The System One request: the shape TypeSafe's Jev takes, served here by
// whatever model is loaded. One state, evaluated against named questions
// of three types: a noul is yes or no and answers with the probability
// of yes; a choice picks one named option and answers with a probability
// per option; a score places the state on ordered levels and answers
// with the expected level. Each question is a labeled read of the logits
// after it, so the whole request is one prefill of the state and one of
// each question, the way ScoreMany runs it. A client written against
// Jev's API can point here instead.

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
)

// SystemOneRequest is one call: a state and the questions asked of it.
// State, instructions and criteria may each be a string or any JSON
// value; what is not a string is shown to the model as JSON.
type SystemOneRequest struct {
	State     json.RawMessage              `json:"state"`
	Model     string                       `json:"model,omitempty"`
	Questions map[string]SystemOneQuestion `json:"questions"`
}

// SystemOneQuestion is one question. Criteria depends on the type: a
// noul may describe true and false, a choice maps each option's name to
// its description, and a score lists its levels low to high.
type SystemOneQuestion struct {
	Type         string          `json:"type"`
	Instructions json.RawMessage `json:"instructions"`
	Criteria     json.RawMessage `json:"criteria,omitempty"`
}

// SystemOneAnswer is one question's answer. Which fields are set
// follows Type: noul for a noul; choice, probabilities and confidence
// for a choice; score, legend, probabilities and confidence for a score.
type SystemOneAnswer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// SystemOneResponse answers a request, one entry per question id.
type SystemOneResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]SystemOneAnswer `json:"answers"`
	Usage   SystemOneUsage             `json:"usage"`
}

// SystemOneUsage is what the request cost: the tokens prefilled and the
// label tokens read.
type SystemOneUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// maxScoreLevels is how many levels a score may have, Jev's limit.
const maxScoreLevels = 10

// SystemOne evaluates a request against the loaded model. Every
// question is rendered as a labeled multiple choice and scored by
// ScoreMany with the state shared, so the answers carry the model's
// next-token probabilities over the labels: the yes label's for a noul,
// each option's for a choice, and the expected level for a score.
func (e *Engine) SystemOne(req SystemOneRequest) (*SystemOneResponse, error) {
	if len(req.Questions) == 0 {
		return nil, errors.New("questions is required")
	}
	// Questions go to the model in id order, so a run is reproducible
	// whatever order the JSON object came in.
	ids := make([]string, 0, len(req.Questions))
	for id := range req.Questions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	plans := make([]systemOnePlan, len(ids))
	qs := make([]Question, len(ids))
	for i, id := range ids {
		p, err := planQuestion(req.Questions[id])
		if err != nil {
			return nil, fmt.Errorf("question %q: %v", id, err)
		}
		plans[i] = p
		qs[i] = p.question
	}
	res, err := e.ScoreMany(jsonText(req.State), qs, true)
	if err != nil {
		return nil, err
	}
	out := &SystemOneResponse{
		Model:   "tensai",
		Answers: make(map[string]SystemOneAnswer, len(ids)),
		Usage:   SystemOneUsage{InputTokens: res.PromptTokens, OutputTokens: res.OptionTokens},
	}
	for i, id := range ids {
		out.Answers[id] = plans[i].answer(res.Probs[i])
	}
	return out, nil
}

// systemOnePlan is a question rendered for the model, with what its
// probabilities mean once they come back.
type systemOnePlan struct {
	typ      string
	question Question
	names    []string // what each option is called in the answer
	legend   map[string]string
}

// planQuestion turns a question into the labeled options the model
// sees. An option shows as its name, and its description after a colon
// when there is one, so the model reads "billing: Payments, invoicing,
// refunds" and answers with the letter.
func planQuestion(q SystemOneQuestion) (systemOnePlan, error) {
	p := systemOnePlan{typ: q.Type}
	text := jsonText(q.Instructions)
	if text == "" {
		return p, errors.New("instructions is required")
	}
	var options []string
	switch q.Type {
	case "noul":
		var crit map[string]json.RawMessage
		if len(q.Criteria) > 0 {
			if err := json.Unmarshal(q.Criteria, &crit); err != nil {
				return p, fmt.Errorf("criteria must be an object with true and false: %v", err)
			}
		}
		p.names = []string{"yes", "no"}
		options = []string{describe("yes", crit["true"]), describe("no", crit["false"])}
	case "choice":
		var crit map[string]json.RawMessage
		if err := json.Unmarshal(q.Criteria, &crit); err != nil || len(crit) < 2 {
			return p, errors.New("criteria must be an object naming at least two options")
		}
		for name := range crit {
			p.names = append(p.names, name)
		}
		sort.Strings(p.names)
		for _, name := range p.names {
			options = append(options, describe(name, crit[name]))
		}
	case "score":
		var levels []json.RawMessage
		if err := json.Unmarshal(q.Criteria, &levels); err != nil || len(levels) < 2 {
			return p, errors.New("criteria must be an array of at least two levels")
		}
		if len(levels) > maxScoreLevels {
			return p, fmt.Errorf("criteria has %d levels, and the most is %d", len(levels), maxScoreLevels)
		}
		p.legend = make(map[string]string, len(levels))
		for i, l := range levels {
			name := strconv.Itoa(i)
			p.names = append(p.names, name)
			p.legend[name] = jsonText(l)
			options = append(options, describe(name, l))
		}
	default:
		return p, fmt.Errorf("type must be noul, choice or score, not %q", q.Type)
	}
	p.question = Question{Text: text, Options: options}
	return p, nil
}

// answer reads a question's probabilities back into its answer type.
func (p systemOnePlan) answer(probs []float64) SystemOneAnswer {
	a := SystemOneAnswer{Type: p.typ}
	switch p.typ {
	case "noul":
		yes := probs[0]
		a.Noul = &yes
		return a
	case "choice":
		a.Probabilities = make(map[string]float64, len(probs))
		best := 0
		for i, name := range p.names {
			a.Probabilities[name] = probs[i]
			if probs[i] > probs[best] {
				best = i
			}
		}
		a.Choice = p.names[best]
	case "score":
		a.Probabilities = make(map[string]float64, len(probs))
		var score float64
		for i, name := range p.names {
			a.Probabilities[name] = probs[i]
			score += float64(i) * probs[i]
		}
		a.Score = &score
		a.Legend = p.legend
	}
	c := confidence(probs)
	a.Confidence = &c
	return a
}

// confidence collapses a distribution into one number from 0 to 1: one
// minus its entropy as a fraction of the most it could have, so all the
// mass on one option is 1 and an even spread is 0. Jev's published
// numbers agree with this to the rounding of its probabilities.
func confidence(probs []float64) float64 {
	if len(probs) < 2 {
		return 1
	}
	var h float64
	for _, p := range probs {
		if p > 0 {
			h -= p * math.Log(p)
		}
	}
	return math.Max(0, 1-h/math.Log(float64(len(probs))))
}

// describe renders an option for the listing: its name, and after a
// colon whatever the criteria said about it.
func describe(name string, criteria json.RawMessage) string {
	if d := jsonText(criteria); d != "" {
		return name + ": " + d
	}
	return name
}

// jsonText is a JSON value as the model should read it: a string as
// itself, anything else as its JSON, and nothing as "".
func jsonText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil || v == nil {
		return ""
	}
	b, _ := json.Marshal(v)
	return string(b)
}
