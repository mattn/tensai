package llm

// The one tool tensai can offer without asking anyone for a key: a
// Wikipedia lookup. It is not web search and does not pretend to be --
// nothing free and keyless indexes the web, and scraping a search engine
// is neither allowed nor stable. What it does do is answer the questions
// a small model most often gets wrong on its own: who or what something
// is, in the words of an encyclopedia rather than its own memory.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"
)

// wikipediaTool is the signature the model is handed.
func wikipediaTool() toolDef {
	return toolDef{Type: "function", Function: toolFunc{
		Name: "wikipedia",
		Description: "Look something up in Wikipedia and read the opening of the best matching article. " +
			"Use it for people, places, organizations, works and technical terms. " +
			"It is an encyclopedia rather than a news source, so it is weak on very recent events.",
		Parameters: json.RawMessage(`{"type":"object","properties":` +
			`{"query":{"type":"string","description":"What to look up, as a title or a short phrase"}},` +
			`"required":["query"]}`),
	}}
}

// wikipediaLookup searches and then reads, so one call is enough: a
// model that had to pick a title from a list and ask again would spend
// two turns to learn what one can tell it.
func wikipediaLookup(ctx context.Context, query string) (string, error) {
	lang := wikiLang(query)
	var search struct {
		Query struct {
			Search []struct {
				Title string `json:"title"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := wikiGet(ctx, lang, url.Values{
		"action":   {"query"},
		"list":     {"search"},
		"srsearch": {query},
		"srlimit":  {"3"},
		"format":   {"json"},
	}, &search); err != nil {
		return "", err
	}
	if len(search.Query.Search) == 0 {
		return fmt.Sprintf("No Wikipedia article matches %q.", query), nil
	}
	titles := make([]string, 0, len(search.Query.Search))
	for _, r := range search.Query.Search {
		titles = append(titles, r.Title)
	}

	var page struct {
		Query struct {
			Pages map[string]struct {
				Title   string `json:"title"`
				Extract string `json:"extract"`
			} `json:"pages"`
		} `json:"query"`
	}
	if err := wikiGet(ctx, lang, url.Values{
		"action":      {"query"},
		"prop":        {"extracts"},
		"exintro":     {"1"},
		"explaintext": {"1"},
		"redirects":   {"1"},
		"titles":      {titles[0]},
		"format":      {"json"},
	}, &page); err != nil {
		return "", err
	}
	var b strings.Builder
	for _, p := range page.Query.Pages {
		fmt.Fprintf(&b, "%s (%s.wikipedia.org)\n%s", p.Title, lang, clipRunes(p.Extract, 700))
	}
	if len(titles) > 1 {
		fmt.Fprintf(&b, "\n\nOther matches: %s", strings.Join(titles[1:], ", "))
	}
	return b.String(), nil
}

// wikiLang picks the edition to ask. A question written in Japanese is
// almost always about something the Japanese Wikipedia covers better,
// and everything else goes to the English one, which is the largest.
func wikiLang(query string) string {
	for _, r := range query {
		if unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Han) {
			return "ja"
		}
	}
	return "en"
}

// wikiGet is one call to the MediaWiki API. Wikimedia asks that a client
// name itself and say where to complain, which is what the user agent is
// for; without one the API answers 403.
func wikiGet(ctx context.Context, lang string, q url.Values, into any) error {
	endpoint := "https://" + lang + ".wikipedia.org/w/api.php?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "tensai (https://github.com/mattn/tensai)")
	resp, err := (&http.Client{Timeout: 15 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("wikipedia: %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	return json.Unmarshal(body, into)
}

// clipRunes cuts to a whole number of runes, at a sentence end when one
// is near, so the model reads a paragraph rather than half a word.
func clipRunes(s string, n int) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) <= n {
		return string(r)
	}
	cut := string(r[:n])
	for _, end := range []string{"。", ". ", ".\n"} {
		if i := strings.LastIndex(cut, end); i > len(cut)/2 {
			return cut[:i+len(end)]
		}
	}
	return cut + "..."
}
