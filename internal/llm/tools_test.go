package llm

import "testing"

func TestResolveTools(t *testing.T) {
	tools, err := resolveTools("wikipedia")
	if err != nil || len(tools) != 1 || tools[0].Function.Name != "wikipedia" {
		t.Fatalf("resolveTools(wikipedia) = %v, %v", tools, err)
	}
	// Nothing offered is not an error: it is what every run without
	// -tool does.
	if tools, err := resolveTools(""); err != nil || tools != nil {
		t.Errorf("resolveTools(empty) = %v, %v; want no tools and no error", tools, err)
	}
	// A typo has to say so. Offering nothing silently would look like a
	// model that declined to call.
	if _, err := resolveTools("wikipdia"); err == nil {
		t.Error("resolveTools(typo) accepted a tool that does not exist")
	}
}

func TestWikiLang(t *testing.T) {
	for q, want := range map[string]string{
		"Linus Torvalds": "en",
		"日本の総理大臣":        "ja",
		"ソフトウェア":         "ja",
		"mattn":          "en",
		"":               "en",
	} {
		if got := wikiLang(q); got != want {
			t.Errorf("wikiLang(%q) = %s, want %s", q, got, want)
		}
	}
}

func TestClipRunes(t *testing.T) {
	// Short enough to keep whole.
	if got := clipRunes("  hello  ", 10); got != "hello" {
		t.Errorf("clipRunes(short) = %q", got)
	}
	// A cut backs up to a sentence end when one is in reach, so the
	// model reads whole sentences.
	long := "A quick note about things. It goes on and on and on and on."
	if got := clipRunes(long, 40); got != "A quick note about things. " {
		t.Errorf("clipRunes(long) = %q, want it cut at the sentence end", got)
	}
	if got := clipRunes("あいうえおかきくけこ。さしすせそたちつてと", 15); got != "あいうえおかきくけこ。" {
		t.Errorf("clipRunes(ja) = %q, want it cut at the 。", got)
	}
	// With no sentence end near the cut it says it was cut, and never in
	// the middle of a multi-byte rune.
	got := clipRunes("あいうえおかきくけこさしすせそたちつてと", 5)
	if got != "あいうえお..." {
		t.Errorf("clipRunes(no sentence end) = %q", got)
	}
}
