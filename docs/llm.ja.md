# LLM 推論

XOR を学習するのと同じカーネルが本物の言語モデルを動かします。`_example/gpt2` は純 Go の完全な推論エンジンで、`tensai` コマンドは同じカーネルの上で 10 のモデルファミリーを動かします。

## GPT-2

`_example/gpt2` は公開されている GPT-2 small (124M) チェックポイントを Hugging Face からダウンロードし、`encoding/safetensors` で重みをロードし、自前のバイトレベル BPE でトークナイズし、KV キャッシュ付きでデコードします — すべての matvec が tensai の他の部分と同じ `Dot` カーネルで走り、AVX2 ビルドで約 30 tok/s:

```
$ GOEXPERIMENT=simd go run ./_example/gpt2 -n 20
Hello, I'm a language model, not a programming language. I'm a language model. ...
```

greedy の続きは GPT-2 のよく知られたリファレンス出力とトークン単位で一致します。リーダー、トークナイザ、順伝播のパイプライン全体が 1 つのチェックで固定されるわけです。

- `-q8` はデコードパスの重みを int8 に量子化して生成を 2 倍にします (同一マシンで 23 → 46 tok/s)。デコードはトークンごとにチェックポイント全体をストリームするからです
- `-gpu` (`-tags wgpu` または `wgpu24` でビルド) は各ブロックの causal マルチヘッド attention を GPU 上の 1 回のマスク付きディスパッチとして実行します

## Qwen とその仲間たち: 10 のモデルファミリー

`tensai` コマンドは現代の instruction-tuned モデルを動かします: RMSNorm、RoPE、grouped-query attention、SwiGLU MLP。safetensors から (config.json が次元を決め、シャーディングされたチェックポイントは index.json 経由) でも、config・トークナイザ・重みを 1 ファイルに収めた llama.cpp の GGUF からでもロードできます。1 つのランタイムが 11 のアーキテクチャを話します:

| ファミリー | モデル | 何が加わるか |
|---|---|---|
| qwen2 | Qwen 1.5/2/2.5, Qwen2.5-Coder, R1-Distill-Qwen 系 | attention バイアス |
| qwen3 | Qwen3 dense | ヘッドごとの QK-norm、明示的 head_dim、`-think` |
| qwen3_5 | Qwen3.5 / 3.6 / 3.8 | 4 層に 3 層が gated delta rule、残り 1 層が通常の attention。正規化は 1 + w、RoPE はヘッドの 1/4 だけ回し、クエリが attention 出力のゲートを連れる。大きいものは 1 つの key head を複数の value head で共有する。CPU のみ、`-draft` 不可 |

`qwen3_5` のプレフィルは、モデルの大きさから想像するより高くつきます。delta 層は
状態をトークンごとに引き継ぐので、バッチが効くのは再帰の周りの射影だけで、長い
プロンプトはプレフィルというよりデコードに近くなります。0.8B で AVX2 マシンなら
1 トークンあたり約 10 ミリ秒 — 同規模の qwen3 が 4 ミリ秒なので、システムプロンプトが
数千トークンあると待たされます。アーキテクチャが許すチャンク化を実装すれば大半は
縮みますが、まだ入っていません。
| llama | Llama 2/3, SmolLM2, Mistral, R1-Distill-Llama | みんながフォークしたブロック |
| smollm3 | SmolLM3-3B | 4 層ごとに RoPE をスキップ |
| gemma3 | Gemma 3 | 5/6 層のスライディングウィンドウ、サンドイッチ norm、gelu-tanh ゲート、SentencePiece |
| gemma4 | Gemma 4 E2B/E4B/12b | トークンごとにディスクから読む per-layer embedding、2 種類のヘッド幅、深い層は前の層の KV キャッシュに attend、logits は tanh でキャップ |
| phi3 | Phi-3/3.5-mini | q/k/v と gate/up が融合済みで配布 |
| qwen2moe / qwen3moe | Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B | top-k ルーティングのエキスパート、qwen2moe は共有エキスパートも |
| gpt-oss | gpt-oss-20b | MXFP4 エキスパート、attention sinks、YaRN rope、harmony チャンネル |
| k2-horizon | K2-Horizon-7B | 行を 4 グループに分けて取る RMSNorm、結合文字と ZWJ を語に含める分割、512K の文脈。GGUF のみ、CPU のみ |

密モデルの 12b はまた別で、per-layer embedding を持たず、KV ヘッド数を層ごとに宣言し (ローカル層 8、グローバル層 1)、狭くなる層には V の射影がありません。その層は K を V として使います。最後の 1 点だけ GPU デコードは対象外です。

Gemma 4 はパラメータの大半 — E2B の 46 億のうち 23 億 — を per-layer
embedding テーブルに置いています。1 ステップで必要なのはそのうち 1 行だけな
ので、テーブルはファイルに残したまま、トークンごとに自分の行だけを読みます。
常駐するのは 2B 相当の普通の transformer です。層は 256 幅のローカルヘッドと
512 幅のグローバルヘッドが交互に並び、深い側の 3 分の 2 は query だけを射影
して同じ種類の最後のキャッシュ層に attend し、logits は最後に tanh でキャップ
されます。GPU デコードも通り、速度は CPU とほぼ同じです。
Radeon 780M の実測で CPU 19.6 tok/s に対して 19.4 tok/s、プレフィルもほぼ同じ。
差が小さいので電源状態の方が効きます。バッテリー駆動では GPU は 5 分の 1 まで
落ちますが、CPU はほとんど変わりません。

DeepSeek-R1 の蒸留モデルに専用ファミリーは要りません — DeepSeek のターンマーカーをまとった素の qwen2/llama ブロックで、ローダーが埋め込みのチャットテンプレートからそれを見つけて自動で切り替えます。`<think>` の推論込みです。

```
$ tensai run -q8 "What is the capital of France?"
The capital of France is Paris.
43 tokens in 1.3s (33.1 tok/s)
```

## 量子化ロード

`-q8` も `-q4` も付けなければローダが幅を自分で選びます。ファイルが持つ精度が上限で (Q4_K のような int4 ブロックは int4 にそのままリパックでき int8 にしても得るものがない、Q8_0 や Q5_K/Q6_K や float のチェックポイントは int8 でないと精度を保てない)、int8 がマシンの空きメモリ (重み + 表ぶん 1/4 + 0.5GB) に収まらないときはスワップするより int4 に落とします。`-v` がどちらをなぜ選んだかを言い、`-f32` は float32 の重みを明示的に頼みます。`-q8`/`-q4` では各重みがロードと同時に量子化され、float32 コピーは即座に破棄されます。フル精度のモデルがメモリに収まる必要はありません。量子化済み GGUF チェックポイントは float32 の回り道を完全にスキップします: Q8_0, Q4_0, Q5_0, Q4_K/Q5_K/Q6_K の K-quant 系、MXFP4、PrismML の三値 PTQ1_0/PQ2_0 がメモリマップしたファイルから直接リパックされ、llama.cpp 自身の量子化がそのまま保たれます。1.5B の Q4_K_M は約 8 秒が約 3 秒に、3B の Q8_0 は 32 秒が 5 秒で開きます (`-requant` は float 経由に戻し、ずっと遅いロードと引き換えにデコードが約 10% 速くなります)。

最初の `.gguf` ロードはリパック済みの重みをモデルの隣のキャッシュファイルに書き (`-nocache` でオプトアウト)、以後のロードはそれをメモリマップするだけです: 1.5B Q4_K_M は約 0.3 秒で、Mistral 7B は 1 秒未満で、gpt-oss-20b は 2 秒未満で再オープンします。マップされた重みはカーネルがいつでも破棄・再読込できるクリーンなファイルバックのページなので、モデルがぎりぎり収まるマシンではスワップのスラッシングが普通のページキャッシュの挙動に置き換わります。

15GB のマシンでの階段はこうなります: 0.5B が `-q8` で約 40 tok/s、1.5B Q4_K_M が `-q4` で約 25 tok/s (タイル化整数カーネル、ネイティブ Windows)、そして Qwen2.5-**7B**-Instruct — 15GB の BF16 シャードを 2 分のロード中にオンザフライで int4 量子化して常駐約 6GB に — が 3.5 tok/s で正しく答えます。

### 三値の重み

PrismML の Bonsai (`Ternary-Bonsai-2-27B`、Qwen3.8-27B ベース) は全重みを -1, 0, +1 の
三値で持ち、128 個ごとに f16 のスケールを 1 つ添えます。エンコードは独自の 2 種で、素の
llama.cpp は読めません: `PTQ1_0` は trit を 1 バイトに 5 つ詰め (27B で 5.95 GB)、`PQ2_0` は
2 ビットのスロットに 1 つずつ置きます (7.21 GB)。どちらも 1 重み 2 ビットの三値レイアウトに
リパックされ、27B が 8 GB 未満でデコードできます。幅の選択はありません。`-q8` や `-q4` は
受け付けますが無視されます (量子化するものがないので)。

```bash
tensai run -model prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf "What is the capital of France?"
```

重みは回転した基底に置かれています。各行列は丸める前に、入力次元に沿ってブロック単位の
Walsh-Hadamard 変換 (固定の符号反転つき) を掛けられていて、これが活性のエネルギーをブロック
全体に均し、3 段階で足りる理由です。ファイルは `prism.hadamard.*` でそれを宣言し、ローダは
その行列が読む活性すべてに同じ変換を、引いた埋め込み行には逆変換を適用します。ローダの知らない
変換を宣言するファイルは、ノイズを吐く代わりに拒否されます。埋め込みテーブルはファイルに
置いたまま 1 行ずつ読みます。回転済みの 25 万行を展開すると数 GB になるからです。

三値カーネルはコードを積和の符号なし側、活性を符号つき側として読むので、必要な補正は
グループの活性の和 1 つで、全列に共通です。列ごとの補正表を重みの隣に流す必要がありません。
Ryzen 7735HS で 27B はプレフィル約 6 tok/s、デコード 3.4 tok/s で、これはメモリ帯域そのもの
です (1 トークンあたり約 28 GB/s の重み読み)。試したプロンプトでは PrismML の llama.cpp ビルドと
トークン単位で一致します。初回ロードは 270 億個の重みのリパックに 1 分ほどかけてリパック
キャッシュ (モデルの隣に 8.5 GB) を書き、以後のロードは 1 秒未満でそれをマップします。
他の qwen3_5 と同じく CPU で走ります。

## プレフィル、投機的デコード、サンプリング

- **バッチプレフィル** — プロンプトは 8 トークン行のブロックでモデルを通り、トークンごとではなくブロックごとに重みを 1 回ストリームするので、最初のトークンまでの待ちが約 6 分の 1 になります
- **投機的デコード** — `-draft` に同系統の小さいモデルを指定します (greedy のみ): ドラフトが数トークン提案し、大きいモデルの 1 回のバッチパスが検証し、却下ならキャッシュをロールバックします。出力は大きいモデル単独とまったく同じです
- **サンプリング** — `-temp` が 0 より大きいと nucleus からサンプリングします: `-topp 0.9` は確率順で 90% の質量を持つ最小のトークン集合だけを残すので、繰り返しループの住処であるロングテールにくじが回りません
- **繰り返しペナルティ** — `-frequency` と `-presence` は OpenAI 流で、この生成で出したトークンは出現 1 回ごとに `-frequency`、出現していれば一律に `-presence` だけ logit を下げます。小さいモデルが同じ段落を繰り返し始めたときに効くのはこちらです。`-repeat` は llama.cpp 流で、直近 `-repeat-last` 位置 (プロンプト込み) に現れたトークンの logit をその値で割ります (1.1 が軽め、1 で無効)。3 つともサンプリング前の logits を変えるので greedy にも効きます。`-repeat` がプロンプトを数える点は諸刃で、プロンプトの言語ごと罰するため、日本語で答えていた 7B が中国語に流れることがあります。`-frequency` は言語に触りません。API では `frequency_penalty`、`presence_penalty`、`repetition_penalty` として同じものを受けます。`-draft` の下ではどれも効きません

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

モデルを使うコマンドは同じフラグを共有します: `-model` (どのモデルを実行するか)、`-q8`/`-q4`/`-f32` (重みの幅。指定がなければファイルとメモリから選ぶ)、`-gpu`、`-draft`、`-think`、`-tool`、`-system`、`-temp`、`-topp`、`-seed` など — 完全なリストは `tensai <command> -h` で。

`-v` は、黙って待つだけだった時間に何をしているかを喋らせます。ファイルが名乗る内容 (アーキテクチャ、層数とヘッド数、コンテキスト、語彙)、重みの読み方 (repack したのかキャッシュから mmap したのか、それぞれ何秒かかったか)、選ばれたテンプレートファミリーとシステムプロンプト、そして他の手段では見えない**実際に組み上がったプロンプト**をマーカーごと出します。`serve` ではリクエストが着いた時点でメッセージ数とツール数を報告し、続けてプレフィルしたトークン数と速度を出します。行が増えるだけで、他は何も変わりません。

システムプロンプトを指定しなければ、ファミリーごとの既定が入ります。`-system ""` は空のシステムターンではなく**システムターンそのものを送りません** — システムメッセージを渡されなかったときにモデル自身のテンプレートが書くのはこの形です。他のランタイムと出力を突き合わせるときは、プロンプトで唯一違うのがここなので、このフラグを使ってください。

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

ゲート付きリポジトリ (Gemma や Llama など) は、ライセンスに同意してトークンを
送るまで 401 を返します。tensai は `HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN`、
`HUGGINGFACE_TOKEN` の順に環境変数を見て、次に `huggingface-cli login` が書く
ファイル (`$HF_HOME/token`、既定では `~/.cache/huggingface/token`) を見ます。
すでにログイン済みのマシンなら追加の設定は要りません。拒否されたダウンロードは
リトライせず、トークンが無いのか、トークンの持ち主がそのリポジトリのライセンスに
同意していないのか、どちらなのかを伝えます。

途中で切れたダウンロードは捨てずに再開します。15GB のファイルでは効きます —
途中まで落ちたものはモデルの隣に `<name>.tmp` として残り、次の試行が続きから
要求します。再開は一緒に記録した ETag で守られているので、上流で差し替わった
チェックポイントは 2 つのバージョンを繋ぎ合わせるのではなくやり直します。
通信エラーとサーバの 5xx は待ち時間を広げながら数回リトライし、404 はしません。

```bash
tensai run -q8 "What is the capital of France?"
tensai run -q8 -model Qwen3-0.6B "Explain RoPE briefly"   # "tensai models" の名前をそのまま
tensai run -q8 -json "Explain RoPE briefly"      # 補完と使用量を 1 つの JSON で
tensai chat -q8 -model ./model.gguf              # マルチターン。KV キャッシュが対話全体を運ぶ
tensai models                                    # キャッシュ一覧。"models rm <name>" で削除
tensai bench -q8                                 # CPU vs GPU のプレフィル/デコード比較
tensai ask -q8 -yesno "Is Paris in France?"      # 生成せず確率で答える
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

### 生成せずに答える

`tensai ask` は生成ではなく計測で答えます。質問は `run` と同じチャットテンプレートを
通り、各選択肢は「モデルがその選択肢を答えの書き出しとして書く対数尤度」として採点され、
その softmax が答えです。トークンは一切サンプリングしないので、モデルは選択肢の外の
ものを答えられず、知らないことは自信ありげな作り話ではなく選択肢間の確率の散らばりとして
現れます。

```bash
tensai ask -q8 -yesno "Is Paris the capital of France? Answer yes or no."
tensai ask -q8 -choice "positive,negative,neutral" "Sentiment of: 'cold food, rude waiter'. One word."
tensai ask -q8 -state "仕事終わり" -choice "コーヒー,ビール,紅茶" "いま何を飲む？ 一語で答えて。"
tensai ask -q8 -json -choice "spam,ham" "Classify: 'You have won a prize'. One word."
```

```
 99.9%  yes
  0.1%  no
```

`-state` は質問の前提となる状況で、ユーザーターンの先頭に置かれます。`-json` は選ばれた
選択肢と各選択肢の確率を返すので、型付きの質問をして型付きの答えを受け取りたい呼び出し元
向けです。コストはプレフィル 1 回と選択肢のトークン数ぶんの decode step で、0.5B なら
数十ミリ秒。プロンプトのキャッシュは選択肢ごとに巻き戻して使い回します。

注意が 2 つ。選択肢は与えた表記のまま採点されます。`yes` と `Yes` は別のトークンで、
モデルがどちらを書きたがるかはモデルの性質なので、「Answer yes or no.」で終わる質問文には
意味があります。もう 1 つ、数値はキャリブレーション込みでモデルのものです。7B に「パリは
ドイツの首都か」と聞くと yes に 20% 置くことがあるので、選択肢間の差を信号として読み、
絶対値はモデルの癖を踏まえて読んでください。知らないことを聞かれた同じ 7B は、`run` では
経歴を自信ありげに捏造しますが、ここでは候補すべてを 50% 付近に置きます。それが散文では
言えない正直な答えです。

#### 1 つの状況に型つきの質問をする

分類器は同じ状況についていくつも聞きます。メッセージは急ぎか、どのチームの担当か、
書き手はどれくらい怒っているか。`-batch` はそれらを TypeSafe の Jev API と同じ形の
1 つのリクエストとして標準入力から受け取り、同じ形で答えます:

```bash
tensai ask -q8 -batch -json <<'EOF'
{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
EOF
```

```json
{"model":"tensai","answers":{
  "is_urgent":   {"type":"noul","noul":0.93},
  "department":  {"type":"choice","choice":"technical","probabilities":{"billing":0.22,"sales":0.12,"technical":0.66},"confidence":0.21},
  "frustration": {"type":"score","score":0.93,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.07,"1":0.93,"2":0.00},"confidence":0.76}},
 "usage":{"input_tokens":174,"output_tokens":8}}
```

質問は 3 種類です。`noul` は yes/no で、yes の確率を返します。`criteria` で yes と no
の意味を補足できます。`choice` は `criteria` に選択肢の名前と説明を並べ、選ばれた名前と
選択肢ごとの確率、confidence を返します。`score` は `criteria` に順序つきのレベルを低い方
から最大 10 個並べ、期待値としてのレベル (2 つのレベルの間の小数になりえます) と凡例、
分布を返します。`confidence` は分布のエントロピーを最大値で割って 1 から引いたもので、
Jev が公開している数値を再現します。`state`、`instructions`、各 criteria は文字列でも
任意の JSON でもよく、文字列でないものは JSON のままモデルに見せます。

内部ではどの質問も A, B, C と文字を振った多肢選択に描画して文字を採点するので、説明が
どれだけ長くても選択肢 1 つは 1 トークンで、答えは質問直後の logits を 1 回読むだけです。
状況は 1 回だけプレフィルされ、各質問はそのキャッシュを延長するので、N 問のコストは
状況 1 回と各質問 1 回ぶんで、状況を N 回読み直しません。同じ描画は 1 問の `ask` でも
`-label` で使えます (なければ選択肢の本文をトークンごとに採点します)。小さいモデルは
質問によらず A に寄るので、0.5B では 1 つの値を鵜呑みにせず選択肢同士を比べてください。

`serve` は同じものを `POST /v1/systemone` として出すので、Jev 向けに書かれた
クライアントを手元のモデルに向けられます。

### OpenAI 互換 API の提供

```bash
tensai serve -q8 -addr 127.0.0.1:8080
```

`models` が一覧するのは `run` / `chat` / `serve` が読めるものだけです —
`config.json` を持つディレクトリか、`.gguf` ファイル。それに加えて、
`tensai image` が使うチェックポイント (`transformer`、`text_encoder`、`vae` を
持つディレクトリ、または ComfyUI 形式のファイル群) も並びます。種別は `diffusers` か `comfyui` で、言語モデルなら `tools` や
`think` が出る列に `image` と出ます。`org/repo` の名前で取得したものは 1 階層下に
置かれ、その名前で並びます。example はデータセットを
同じ場所にキャッシュしますが、それらはモデルとして並べず件数だけ報告します
(`models rm` では名前を指定して削除できます)。

```
Qwen-Image-2.1                             52.2GB  diffusers image       2026-09-22
Qwen/Qwen3-4B-Instruct-2507                 7.5GB  qwen3     tools think 2026-08-27
Qwen2.5-1.5B-Instruct                       2.9GB  qwen2     tools       2026-08-23
SmolLM2-360M-Instruct                       692MB  llama     -           2026-08-24
qwen2.5-0.5b-instruct-q8_0.gguf             531MB  gguf      tools       2026-08-25
```

リポジトリから落としたモデルは、組織名込みでそのリポジトリ名で並びます —
キャッシュのディレクトリ名は組織名を落としてしまうので、それが無いと
「まだ持っていないマシンで何と打てばいいか」を一覧が答えられないからです。
既にキャッシュがあるマシンではどちらの形でも通り、`models rm` も両方受けます。
これ以前にキャッシュしたものや手で置いたものは、次にダウンロードされるまで
素のディレクトリ名のままです。

4 列目が言うのは `serve` がそのモデルをどう扱うか — `tools` 付きのリクエストを
受けるか、`-think` に思考ブロックを与えるか — であって、**どれだけ上手いかでは
ありません**。0.5B が tools と出るのは、ツールを提示されるからであって、
呼び出しが信頼できるからではありません。判定はローダーと完全に同じ手順
(ファミリーへのフォールバック込み) なので、自分のテンプレートがディスクに無い
チェックポイントも、実際に扱われるとおりに並びます。読み取りコストは `.gguf`
1 つあたり約 80ms のメタデータ解析で、ディレクトリはタダです。

`serve` は `/v1/chat/completions` (messages 配列、SSE ストリーミング、使用量カウント) を公開するので、OpenAI クライアントを向ければ何でも純 Go のモデルとチャットできます。`ask -batch` の型つき質問を HTTP で受ける `/v1/systemone` もあります。組み込みのチャットデモページが `GET /` で提供されます。

### 思考の分離

`-think` を付けると、答える前に考えるモデルの出力は 2 つに分かれて届きます — 最初に書く思考ブロックが `reasoning_content`、返答だけが `content` で、ストリーミングでもそれぞれのデルタになります。クライアントは思考を思考として表示することも、捨てることもできます。思考がプロンプトに戻ることはありません — 履歴を再生するときは返答だけを残し、思考は落とします。モデル自身のテンプレートと同じ扱いです。

```json
"message": {
  "role": "assistant",
  "reasoning_content": "Okay, the user is asking for 17 multiplied by 3...",
  "content": "17 multiplied by 3 is 51."
}
```

`-think` なしの場合、qwen3 と smollm3 は空の思考ブロックでターンを開くので、分離するものがありません。gpt-oss は harmony チャンネルで推論する別機構で、これは対象外です。

### ツール呼び出し

`tools` を渡すと、モデルは散文の代わりに `tool_calls` で答えられます。エージェントがループを回すのに必要なものです:

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}'
```

```json
"finish_reason": "tool_calls",
"message": {"role": "assistant", "content": "", "tool_calls": [
  {"index": 0, "id": "call_0", "type": "function",
   "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}]}
```

結果は、呼び出した assistant ターンと一緒に、対応する呼び出しを名指しした `tool` メッセージとして返します。するとモデルが返答を書きます。ストリーミングでも同じで、テキストは通常どおり流れ、呼び出しは 0 始まりの index を持つ `tool_calls` デルタとして届き、ターンは `finish_reason: "tool_calls"` で終わります。

シグネチャは、そのファミリー自身が訓練された形式でモデルに渡されます。ChatML 系 — qwen2、qwen3、それぞれの MoE、llama、SmolLM — では、システムターンの末尾に追加される `<tools>` ブロックと、`<tool_call>` JSON による呼び出しです。Qwen3.5 はこれを離れており、`<tools>` ブロックがシステムターンの先頭に置かれ、呼び出しは引数ごとの `<parameter=key>` を持つ `<function=name>` 要素になります。この形式は型を運ばないので、値の意味はツール自身の JSON Schema が決めます。個々のチェックポイントがツール用に用意されているかどうかは推測ではありません — そのモデル自身のチャットテンプレートが、渡されうるツール定義を分岐で受けるか受けないかで決まり、その答えがファミリー推定に優先します。GGUF はテンプレートをメタデータに埋め込んでおり、ダウンロードしたチェックポイントは `tokenizer_config.json` (新しいものはそこが空で `chat_template.jinja` に置く) から読みます。パスやキャッシュ名で指定したモデルはその場で読むだけで取りに行かないので、この機能より前にキャッシュしたものはファイルが揃うまでファミリー推定にフォールバックします。Gemma 4 は 4 つ目の慣習を持っています。JSON でも XML でもなく、文字列を引用符ではなく `<|"|>` で囲む波括弧の DSL です。シグネチャはシステムターンに `<|tool>declaration:name{...}<tool|>` として置かれ、呼び出しは `<|tool_call>call:name{key:value}<tool_call|>` で返り、それに答える結果は**同じ model ターンの中**に `<|tool_response>response:name{value:...}<tool_response|>` として戻します。モデルは新しいターンを開かずそのターンの続きを書きます。続きを書くにはもう 1 つ必要なものがあります。gemma4 が毎ターンの先頭で自分から開く thought チャンネルです。ここではターンが既に始まっているので自分では開けず、プロンプト側からマーカーを渡します。これが無いと E4B はツール結果を受け取った直後に停止し、渡すと同じプロンプトに答えます。続く思考はいつもどおり分離されるので、誰に見せるかは `-think` と `reasoning_content` が決めます。小さいモデルは温度付きサンプリングで呼び出し本体だけ書いてマーカーを忘れることがあるので、呼び出し側が宣言した名前で始まる `name{...}` は呼び出しとして読みます。慣習を持たないファミリー (gemma3、phi3、mistral、deepseek、gpt-oss) と、テンプレートがツールに触れないチェックポイント (例えば SmolLM2) は、`tools` 付きのリクエストを黙って無視せず 400 で返します。サンプラーは拘束されないので、呼ぶかどうかはモデル次第です。`tool_choice: "none"` はシグネチャを渡しませんが、`"required"` は文法拘束でしか実現できないため `"auto"` として扱われます。既定の 0.5B より大きいモデルのほうが、はるかに確実に呼び出します。

- デフォルトのバインドはループバックのみです (`127.0.0.1:8080`、または `$TENSAI_ADDR`)。広げるときは明示的にどうぞ
- `-api-key` (または `$TENSAI_API_KEY`) は `/v1` ルートに bearer トークンを要求します。デモページは開いたままです
- エージェントが毎ターン送り直すもの — システムメッセージとツール定義 — は
  一度だけプレフィルして保持します。直前の続きならそのまま継続し、冒頭だけが
  共通なら 2 つが分かれた地点から再開します (その地点は、最初に分岐を見たときに
  チェックポイントとして控えます)。744 トークンのプロンプトで、初回 12 秒・
  以降 2 秒未満になります。GPU パスは自前の常駐キャッシュを持つので対象外です

### サーバーなしでツールを使う

`serve` はクライアントがツールを持ってくるのを待ちますが、`run` と `chat` は
自分で 1 つ持てます。`-tool wikipedia` を渡すとモデルに Wikipedia の検索が
提供され、モデルが呼び出したらその場で実行し、結果を会話に戻し、モデルは
読んだ内容から答えます。

```bash
tensai run -q4 -model unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf \
  -tool wikipedia -n 400 "Who is Linus Torvalds?"
```

```
Linus Torvalds is a Finnish and American software engineer, best known as the
creator and lead developer of the Linux kernel since 1991.
```

呼び出しは読み手ではなくツールに向けた発話なので、標準出力には流しません。
`-json` も答えだけを返します。ツール結果は呼び出したターンの中に
書き戻す必要があるため、各ラウンドで会話全体を描き直してプレフィルし直します。
モデルが呼び出せるのは最大 4 往復までで、その後は答える必要があります。`-n`
には余裕を持たせてください。思考するモデルは、呼び出しに、結果を読むのに、
そして最後に返答にトークンを使います。

`wikipedia` は、キーもアカウントも要らない唯一のツールです。検索してから
最良一致の冒頭を読むので、1 回の呼び出しで済み (2 往復になりません)、モデルが
別の記事を狙っていた場合に備えて他の候補タイトルも一緒に返します。日本語で
書かれたクエリは `ja.wikipedia.org`、それ以外は `en.wikipedia.org` を引きます。
これは検索エンジンではなく百科事典です。何者か・何であるかには答えますが、
今週のニュースには弱いままです。Wikimedia はクライアントが名乗ることを求めて
おり、`tensai` の User-Agent がそれにあたります。プロキシがそれを落とす環境では
403 がツールの答えとしてモデルに渡り、モデルはそう言えます。
