# トークナイザ

`tokenizer` パッケージは Hugging Face の `tokenizer.json` を読み込み、バイトレベル BPE 系と SentencePiece の Unigram モデルを実装します。GGUF の語彙から組み立てる SentencePiece もあります。

```go
import "github.com/mattn/tensai/tokenizer"

tok, err := tokenizer.Load("tokenizer.json") // モデルが Hugging Face で配布するファイル
ids := tok.Encode("Hello, I'm a language model,")
text := tok.Decode(ids)
eos, _ := tok.ID("<|endoftext|>")
```

## バイトレベル BPE

GPT-2、Llama 3、Qwen が使う形のバイトレベル BPE です。これらのモデルが宣言する事前トークン化の正規表現には、Go の `regexp` では表現できない先読みやインラインの大文字小文字無視グループが必要なので、実世界に存在する分割パターンは手書きのスキャナとして実装されています:

- **GPT-2** の分割
- **cl100k** スタイルの分割
- **o200k** の分割 (gpt-4o / gpt-oss)

それ以外は黙って誤トークン化する代わりに拒否されます。特殊トークンはエンコード時にそのまま照合されます。NFC 正規化はパススルーです — 入力はすでに NFC であると仮定します。実世界のテキストはほぼすべてそうです。

## SentencePiece

`NewSPM` は GGUF の語彙から SentencePiece トークナイザを組み立てます — Gemma や Llama-2 世代のモデル用です。

## Unigram

モデルが `Unigram` の `tokenizer.json` (LLM-jp のもの) は、語彙ごとに対数確率を持ちます。テキストは、スコアの合計が最大になる分割で区切ります。全分割を対象にしたビタビ探索で、同じ語彙でもスコアの高いペアを貪欲に結合する方式とは結果が変わります。どの語彙にも含まれない文字は、sentencepiece や `tokenizers` と同じく最低スコアから 10 引いたコストで扱い、UTF-8 のバイトトークンとして出します。理解する正規化はこの系統の 2 つです。1 つは LLM-jp のもので、特殊トークンで区切った各区間の先頭と空白を U+2581 にします。もう 1 つは Ruri v3 の Metaspace プリトークナイザ (`prepend_scheme` が `never`) で、空白は置き換えますが先頭には何も付けません。それ以外は拒否します。デコードでは、先頭と特殊トークンの直後に付いた空白を、LLM-jp 自身のトークナイザと同じく落とします。`DecodeNext` は、1 トークンずつのストリームを全体と同じ文字列にデコードします。エンコード側の対が `EncodeAfter` です。普通のトークンに続くテキストには区間の印を付けないので、ターンごとに入力する chat でも、会話全体をまとめてエンコードしたときと同じトークン列になります。`NewUnigram` は、GGUF ファイルの語彙、スコア、トークン種別から同じトークナイザを組み立てます。制御トークンはそのまま照合し、バイトトークンをフォールバックに使います。Ruri v3 の GGUF はこれを通り、transformers と完全に同じトークン列になります。

## 検証

エンコード結果はリファレンスの `tokenizers` ライブラリと `llama-tokenize` に対して検証されています: 敵対的コーパスと 2000 のファズ文字列が GPT-2 と Qwen2.5 の両方でエンコード・デコードとも完全一致します (`tokenizer/verify_hf.py` 参照)。
