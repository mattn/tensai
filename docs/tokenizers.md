# Tokenizers

The `tokenizer` package loads Hugging Face `tokenizer.json` files and implements the byte-level BPE family and the SentencePiece Unigram model, plus SentencePiece built from GGUF vocabularies.

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

## Unigram

A `tokenizer.json` whose model is `Unigram` (LLM-jp's) keeps a log probability per piece, and a text splits into the pieces whose scores sum highest: a Viterbi search over every split, which gives different tokens from merging the best-scoring pair greedily on the same vocabulary. A character no piece covers costs the lowest score less ten, as in sentencepiece and `tokenizers`, and comes out as its UTF-8 byte tokens. The normalizer understood is the one this family ships, a U+2581 in front of every segment between special tokens and in place of every space; any other is refused. Decoding takes that leading space back off at the start and after a special token, the way LLM-jp's own tokenizer does, and `DecodeNext` decodes a stream of single tokens to the same text. `EncodeAfter` is the encoding twin: text that continues the sequence after an ordinary token gets no segment mark, so a chat fed turn by turn sees the tokens of the whole conversation.

## Verification

Encodings are verified against the reference `tokenizers` library and `llama-tokenize`: an adversarial corpus and 2000 fuzzed strings encode and decode identically for both GPT-2 and Qwen2.5 (see `tokenizer/verify_hf.py`).
