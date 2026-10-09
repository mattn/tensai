# Text Embeddings

`tensai embed` turns texts into vectors with an embedding model, and `tensai serve` answers OpenAI's `/v1/embeddings` with one, which is what a chat front end's document search (RAG) asks for. The model runs on the same pure-Go kernels as the rest. The default is [Ruri v3 310m](https://huggingface.co/cl-nagoya/ruri-v3-310m), a Japanese ModernBERT, in the [imatrix Q4_K_M conversion](https://huggingface.co/Targoyle/ruri-v3-310m-GGUF-Q4_K_M-imatrix) (204MB, downloaded on first use).

```
$ tensai embed -sim "川べりでサーフボードを持った人たちがいます" "サーファーたちが川べりに立っています" \
    "トピック: 瑠璃色のサーファー" "検索クエリ: 瑠璃色はどんな色？" \
    "検索文書: 瑠璃色（るりいろ）は、紫みを帯びた濃い青。名は、半貴石の瑠璃（ラピスラズリ、英: lapis lazuli）による。JIS慣用色名では「こい紫みの青」（略号 dp-pB）と定義している[1][2]。"
1.0000 0.9573 0.8144 0.7054 0.6891
0.9573 1.0000 0.8211 0.7012 0.6800
0.8144 0.8211 1.0000 0.8715 0.8475
0.7054 0.7012 0.8715 1.0000 0.9729
0.6891 0.6800 0.8475 0.9729 1.0000
```

That is the example on Ruri's model card, and every entry is within 0.003 of the table printed there from the original float32 weights.

Without `-sim` each text comes out as one JSON line, `{"index":0,"tokens":11,"embedding":[...]}`. Texts are the arguments, or one per line on stdin when there are none. Every vector has unit length, so a dot product is the cosine similarity.

Ruri v3 is trained with a prefix that says what a text is for, and expects it to be part of the text: `検索クエリ: ` for a search query, `検索文書: ` for a document to be found, `トピック: ` for classification and clustering, and nothing for plain meaning. tensai passes texts through as they are, so the prefix is the caller's to add, the same as with sentence-transformers or llama.cpp.

## What runs

ModernBERT is an encoder: every token sees every other, in both directions, and the result is one vector per token rather than a next token.

1. **Tokens.** Ruri's vocabulary is a SentencePiece Unigram model, which tensai searches as one (the best-scoring split) and frames with `<s>` and `</s>`. The file does not say whether a space goes in front of the text, and llama.cpp assumes one; the `tokenizer.json` it was converted from adds none, and neither does tensai, so the ids match transformers exactly on every text tried.
2. **The encoder.** 25 layers of LayerNorm (no bias), attention with RoPE over 12 heads, and a GeGLU feed-forward. The first layer and every third after it see the whole text with a RoPE base of 160000; the others see only the 64 tokens to either side, with a base of 10000.
3. **Pooling.** The final hidden states are averaged over the text and scaled to unit length.

Several texts share one pass over the weights (up to about 2048 tokens of them), each with its own positions and attending only within itself, so a batch gives each text the vector it would get alone.

## Accuracy and width

Against the original float32 model in transformers, on seven texts from 8 to 382 tokens (cosine of the two vectors):

| | lowest | highest |
|---|---|---|
| tensai, float32 (default) | 0.9935 | 0.9981 |
| llama.cpp b10615, same file and ids | 0.9936 | 0.9980 |
| tensai `-q8` | 0.9912 | 0.9945 |
| tensai `-q4` | 0.9877 | 0.9926 |

What is left at float32 is the Q4_K_M file's own quantization, the same for llama.cpp. The narrower widths keep the stored blocks as they are (Q4_K and Q6_K repack directly), but the integer matmuls also round each activation row to 7 bits against one scale, and an encoder's rows carry outliers that make that the larger error. That is why float32 is the default here, unlike `run`.

Speed and memory on a Ryzen 7 7735HS (16 threads), loading in about 1.7s:

| width | tokens/s | peak memory |
|---|---|---|
| float32 | 240 to 255 | 2.1GB |
| `-q8` | 420 to 450 | 0.9GB |

Nearly all of the float32 time is the matrix multiply itself.

## Serving

```bash
tensai serve -model Targoyle/ruri-v3-310m-GGUF-Q4_K_M-imatrix/ruri-v3-310m-Q4_K_M-imatrix.gguf
```

An embedding model named by `-model` makes a server with `/v1/embeddings` and `/v1/models` only. To have one server do both, give the chat model as usual and the embedding model with `-embed`:

```bash
tensai serve -q8 -embed ruri-v3-310m-Q4_K_M-imatrix
```

`/v1/embeddings` takes OpenAI's request: `input` as a string, an array of strings, an array of token ids, or an array of those. `encoding_format` is `float` or `base64` (little-endian float32), and OpenAI's own client libraries ask for base64 by default. `usage` counts the tokens read, the framing ones included. Embedding requests keep no state, so they run alongside each other and alongside a chat request.

```bash
curl -s localhost:8080/v1/embeddings -d '{"input": ["検索クエリ: 瑠璃色はどんな色？", "検索文書: 瑠璃色は、紫みを帯びた濃い青。"]}'
```

`tensai models` lists an embedding model with `embed` in the column that says `tools` or `think` for a language model, and `run` or `chat` on one says to use `embed` or `serve` instead.

## Flags

```
tensai embed [flags] [text ...]
```

`-model` (named the way `run` takes one: a cached name, a `.gguf` path, or `org/repo/file.gguf`), `-q8`/`-q4`, `-sim`, and `-v`. Only GGUF files of the `modern-bert` architecture are read.
