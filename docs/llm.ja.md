# LLM 推論

XOR を学習するのと同じカーネルが本物の言語モデルを動かします。`_example/gpt2` は純 Go の完全な推論エンジンで、`tensai` コマンドは同じカーネルの上で 9 つのモデルファミリーを動かします。

## GPT-2

`_example/gpt2` は公開されている GPT-2 small (124M) チェックポイントを Hugging Face からダウンロードし、`encoding/safetensors` で重みをロードし、自前のバイトレベル BPE でトークナイズし、KV キャッシュ付きでデコードします — すべての matvec が tensai の他の部分と同じ `Dot` カーネルで走り、AVX2 ビルドで約 30 tok/s:

```
$ GOEXPERIMENT=simd go run ./_example/gpt2 -n 20
Hello, I'm a language model, not a programming language. I'm a language model. ...
```

greedy の続きは GPT-2 のよく知られたリファレンス出力とトークン単位で一致します。リーダー、トークナイザ、順伝播のパイプライン全体が 1 つのチェックで固定されるわけです。

- `-q8` はデコードパスの重みを int8 に量子化して生成を 2 倍にします (同一マシンで 23 → 46 tok/s)。デコードはトークンごとにチェックポイント全体をストリームするからです
- `-gpu` (`-tags wgpu` または `wgpu24` でビルド) は各ブロックの causal マルチヘッド attention を GPU 上の 1 回のマスク付きディスパッチとして実行します

## Qwen とその仲間たち: 9 つのモデルファミリー

`tensai` コマンドは現代の instruction-tuned モデルを動かします: RMSNorm、RoPE、grouped-query attention、SwiGLU MLP。safetensors から (config.json が次元を決め、シャーディングされたチェックポイントは index.json 経由) でも、config・トークナイザ・重みを 1 ファイルに収めた llama.cpp の GGUF からでもロードできます。1 つのランタイムが 9 つのアーキテクチャを話します:

| ファミリー | モデル | 何が加わるか |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, R1-Distill-Qwen 系 | attention バイアス |
| qwen3 | Qwen3 dense | ヘッドごとの QK-norm、明示的 head_dim、`-think` |
| llama | Llama 2/3, SmolLM2, Mistral, R1-Distill-Llama | みんながフォークしたブロック |
| smollm3 | SmolLM3-3B | 4 層ごとに RoPE をスキップ |
| gemma3 | Gemma 3 | 5/6 層のスライディングウィンドウ、サンドイッチ norm、gelu-tanh ゲート、SentencePiece |
| phi3 | Phi-3/3.5-mini | q/k/v と gate/up が融合済みで配布 |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k ルーティングのエキスパート、qwen2moe は共有エキスパートも |
| gpt-oss | gpt-oss-20b | MXFP4 エキスパート、attention sinks、YaRN rope、harmony チャンネル |

DeepSeek-R1 の蒸留モデルに専用ファミリーは要りません — DeepSeek のターンマーカーをまとった素の qwen2/llama ブロックで、ローダーが埋め込みのチャットテンプレートからそれを見つけて自動で切り替えます。`<think>` の推論込みです。

```
$ tensai run -q8 "What is the capital of France?"
The capital of France is Paris.
43 tokens in 1.3s (33.1 tok/s)
```

## 量子化ロード

`-q8`/`-q4` では各重みがロードと同時に量子化され、float32 コピーは即座に破棄されます。フル精度のモデルがメモリに収まる必要はありません。量子化済み GGUF チェックポイントは float32 の回り道を完全にスキップします: Q8_0, Q4_0, Q5_0, Q4_K/Q5_K/Q6_K の K-quant 系、MXFP4 がメモリマップしたファイルから直接リパックされ、llama.cpp 自身の量子化がそのまま保たれます。1.5B の Q4_K_M は約 8 秒が約 3 秒に、3B の Q8_0 は 32 秒が 5 秒で開きます (`-requant` は float 経由に戻し、ずっと遅いロードと引き換えにデコードが約 10% 速くなります)。

最初の `.gguf` ロードはリパック済みの重みをモデルの隣のキャッシュファイルに書き (`-nocache` でオプトアウト)、以後のロードはそれをメモリマップするだけです: 1.5B Q4_K_M は約 0.3 秒で、Mistral 7B は 1 秒未満で、gpt-oss-20b は 2 秒未満で再オープンします。マップされた重みはカーネルがいつでも破棄・再読込できるクリーンなファイルバックのページなので、モデルがぎりぎり収まるマシンではスワップのスラッシングが普通のページキャッシュの挙動に置き換わります。

15GB のマシンでの階段はこうなります: 0.5B が `-q8` で約 40 tok/s、1.5B Q4_K_M が `-q4` で約 25 tok/s (タイル化整数カーネル、ネイティブ Windows)、そして Qwen2.5-**7B**-Instruct — 15GB の BF16 シャードを 2 分のロード中にオンザフライで int4 量子化して常駐約 6GB に — が 3.5 tok/s で正しく答えます。

## プレフィル、投機的デコード、サンプリング

- **バッチプレフィル** — プロンプトは 8 トークン行のブロックでモデルを通り、トークンごとではなくブロックごとに重みを 1 回ストリームするので、最初のトークンまでの待ちが約 6 分の 1 になります
- **投機的デコード** — `-draft` に同系統の小さいモデルを指定します (greedy のみ): ドラフトが数トークン提案し、大きいモデルの 1 回のバッチパスが検証し、却下ならキャッシュをロールバックします。出力は大きいモデル単独とまったく同じです
- **サンプリング** — `-temp` が 0 より大きいと nucleus からサンプリングします: `-topp 0.9` は確率順で 90% の質量を持つ最小のトークン集合だけを残すので、繰り返しループの住処であるロングテールにくじが回りません

## `tensai` コマンド

```bash
GOEXPERIMENT=simd go install github.com/mattn/tensai/cmd/tensai@latest
```

```
usage: tensai <command> [flags]

commands:
  run      generate a completion for a prompt
  chat     interactive multi-turn chat on stdin
  serve    OpenAI-compatible /v1/chat/completions server
  bench    compare CPU and GPU prefill and decode speed
  models   list cached models; "models rm <name>" deletes one
  version  print the version
```

モデルを使うコマンドは同じフラグを共有します: `-model` (どのモデルを実行するか)、`-q8`/`-q4`、`-gpu`、`-draft`、`-think`、`-system`、`-temp`、`-topp`、`-seed` など — 完全なリストは `tensai <command> -h` で。

どのモデルを実行するかを言うのは `-model` だけで、次の順に解釈します:

| 形式 | 例 |
|---|---|
| `tensai models` が出す名前 | `-model Qwen3-0.6B`、`-model qwen2.5-0.5b-instruct-q8_0` |
| ディレクトリまたは `.gguf` のパス | `-model ./model.gguf`、`-model /srv/checkpoints/qwen` |
| Hugging Face リポジトリ (初回にダウンロード) | `-model Qwen/Qwen3-4B-Instruct-2507` |

省略すると既定のチェックポイントです。ローカル指定が勝手にダウンロードすることは
ありません — キャッシュに無く、取得元の組織名も無い名前はエラーになり、一覧を案内
します。ダウンロードはユーザーキャッシュディレクトリ (Linux なら `~/.cache/tensai`)
に置かれます。それ以外の場所に置きたいなら、そこへ取得してからパスで指定してください。
`-draft` も同じ形式を受けます (`.gguf` を除く)。

```bash
tensai run -q8 "What is the capital of France?"
tensai run -q8 -model Qwen3-0.6B "Explain RoPE briefly"   # "tensai models" の名前をそのまま
tensai run -q8 -json "Explain RoPE briefly"      # 補完と使用量を 1 つの JSON で
tensai chat -q8 -model ./model.gguf              # マルチターン。KV キャッシュが対話全体を運ぶ
tensai models                                    # キャッシュ一覧。"models rm <name>" で削除
tensai bench -q8                                 # CPU vs GPU のプレフィル/デコード比較
```

### CPU と GPU の比較

`bench` は合成プロンプトのプレフィルと数トークンのデコードを CPU と GPU で 1
回ずつ実行し、両方と倍率を表示します。各側は別プロセスで走るので、解放済み
モデルのページがもう一方の測定を汚しません。ヘッダには両側が使っているカーネルと
アダプタが出ます — これが重要で、2 つのバインディング世代は届くアダプタが
異なり (WSL2 の Mesa dozen のような非準拠ドライバが見えるのは `wgpu24`
だけ)、タグを間違えると気付かないまま CPU Vulkan 実装にフォールバックします。
また `GOEXPERIMENT=simd` なしでビルドするとポータブルカーネルを測ることに
なり、AVX2 版より一桁遅い数値が出ます。

```
$ GOEXPERIMENT=simd go run -tags wgpu24 ./cmd/tensai bench -q8
prefill 401 tokens, decode 32 tokens, int8 weights
cpu: AVX2 kernels
gpu: Microsoft Direct3D12 (AMD Radeon(TM) Graphics) (integrated) via -tags wgpu24

median of 5 runs after one warm-up, tokens/sec

           prefill                  decode
cpu          430.7 (357-446)          38.3 (38-39)
gpu         2241.5 (1663-2295)        28.7 (25-29)
gpu/cpu      5.20x                   0.75x
```

`-p` でプロンプトのおおよそのトークン数、`-n` でデコードするトークン数、`-r`
で計測の反復回数を指定します。GPU ビルドタグなしの場合は、GPU 行に理由が
表示されます。反復の間モデルは常駐したままで、最初の 1 回は捨てられるので、
サンプルは定常状態を表します — この経路ではコールドのプレフィルが 3 割ほど
低く出ることがあり、定常状態を報告するツールと比べるとそれが不公平になり
ます。プレフィルの t/s は attention が二次なのでプロンプトが長いほど下がり
ます。比較は同じ長さで行ってください。

### OpenAI 互換 API の提供

```bash
tensai serve -q8 -addr 127.0.0.1:8080
```

`models` が一覧するのは `run` / `chat` / `serve` が読めるものだけです —
`config.json` を持つディレクトリか、`.gguf` ファイル。example はデータセットを
同じ場所にキャッシュしますが、それらはモデルとして並べず件数だけ報告します
(`models rm` では名前を指定して削除できます)。

`serve` は `/v1/chat/completions` (messages 配列、SSE ストリーミング、使用量カウント) を公開するので、OpenAI クライアントを向ければ何でも純 Go のモデルとチャットできます。組み込みのチャットデモページが `GET /` で提供されます。

- デフォルトのバインドはループバックのみです (`127.0.0.1:8080`、または `$TENSAI_ADDR`)。広げるときは明示的にどうぞ
- `-api-key` (または `$TENSAI_API_KEY`) は `/v1` ルートに bearer トークンを要求します。デモページは開いたままです
