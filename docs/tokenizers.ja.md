# トークナイザ

`tokenizer` パッケージは Hugging Face の `tokenizer.json` を読み込み、バイトレベル BPE 系を実装します。GGUF の語彙から組み立てる SentencePiece もあります。

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

## 検証

エンコード結果はリファレンスの `tokenizers` ライブラリと `llama-tokenize` に対して検証されています: 敵対的コーパスと 2000 のファズ文字列が GPT-2 と Qwen2.5 の両方でエンコード・デコードとも完全一致します (`tokenizer/verify_hf.py` 参照)。
