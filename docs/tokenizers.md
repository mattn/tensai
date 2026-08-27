# Tokenizers

The `tokenizer` package loads Hugging Face `tokenizer.json` files and implements the byte-level BPE family, plus SentencePiece built from GGUF vocabularies.

```go
import "github.com/mattn/tensai/tokenizer"

tok, err := tokenizer.Load("tokenizer.json") // the file models ship on Hugging Face
ids := tok.Encode("Hello, I'm a language model,")
text := tok.Decode(ids)
eos, _ := tok.ID("<|endoftext|>")
```

## Byte-level BPE

Byte-level BPE as GPT-2, Llama 3, and Qwen use it. The pre-tokenization regexes these models declare need lookahead and inline case-insensitive groups that Go's `regexp` cannot express, so the split patterns that exist in the wild are hand-written scanners:

- the **GPT-2** split
- the **cl100k**-style split
- the **o200k** split (gpt-4o / gpt-oss)

Anything else is rejected rather than silently mis-tokenized. Special tokens are matched verbatim during encode. An NFC normalizer passes through — input is assumed already NFC, which virtually all real-world text is.

## SentencePiece

`NewSPM` builds a SentencePiece tokenizer from a GGUF vocabulary — the Gemma and Llama-2-era models.

## Verification

Encodings are verified against the reference `tokenizers` library and `llama-tokenize`: an adversarial corpus and 2000 fuzzed strings encode and decode identically for both GPT-2 and Qwen2.5 (see `tokenizer/verify_hf.py`).
