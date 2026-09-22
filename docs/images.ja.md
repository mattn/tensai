# 画像生成

`tensai image` は [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) でプロンプトから絵を描きます。ほかと同じ純粋な Go で、cgo なし、C コンパイラ不要、XOR ネットワークを学習させるのと同じカーネルの上で動きます。

```
$ tensai image -size 256 -steps 20 "a calico cat asleep on a stack of books"
prompt: 18 tokens in 6s
transformer: loaded in 0s
step 20/20 in 3m39s
wrote out.png, 256x256
```

## 何が走るか

3 つのモデルが順に走ります。どれも 7GB あるので、同時に持つのは 1 つだけです。

1. **プロンプトエンコーダ**。denoising transformer はプロンプトをトークンではなく vision-language モデルの隠れ状態として読むので、まず Qwen3-VL 8B の言語側 36 層が走ります。渡すのは最終ノルムを**通す前**の最後の層の出力で、テンプレートのシステム前置きは落とします。終わったら解放します。
2. **denoising transformer**。プロンプトのトークンと画像の latent 格子をつないだ 1 本の列を、単流のブロック 32 層が通ります。1 ブロックは attention と SwiGLU の feed-forward で、挟む層正規化は重みを持ちません。持たないスケールとゲートは timestep から来ます。これを 20 回通すと latent がノイズから絵になります。
3. **デコーダ**。latent の 1 点を 16 ピクセルの正方形にする 2D オートエンコーダです。チェックポイントがそのまま生成するアルファチャンネルも持ちます。

すべてのピースを参照実装と突き合わせてあります (transformer・スケジュール・デコーダは diffusers、プロンプトエンコーダは transformers)。差は 6.5e-8 から 2.4e-4 で、float32 の丸め誤差の範囲です。参照は全層ではメモリに載らないため層数を絞って照合していますが、ブロック以外の配線は層数に依存しません。

## チェックポイントの入手

まだダウンローダはありません。`~/.cache/tensai/Qwen-Image-2.1` の下に自分で置いてください。合計およそ 31GB です。

```bash
cd ~/.cache/tensai/Qwen-Image-2.1
base=https://huggingface.co/Qwen/Qwen-Image-2.1/resolve/main

mkdir -p vae transformer text_encoder processor
curl -L -o processor/tokenizer.json $base/processor/tokenizer.json
curl -L -o vae/config.json $base/vae/config.json
curl -L -o vae/diffusion_pytorch_model.safetensors $base/vae/diffusion_pytorch_model.safetensors
for f in diffusion_pytorch_model.safetensors.index.json \
         diffusion_pytorch_model-00001-of-00002.safetensors \
         diffusion_pytorch_model-00002-of-00002.safetensors; do
  curl -L -o transformer/$f $base/transformer/$f
done
for f in model.safetensors.index.json \
         model-0000{1,2,3,4}-of-00004.safetensors; do
  curl -L -o text_encoder/$f $base/text_encoder/$f
done
```

`-model` はキャッシュ下の名前でも、`text_encoder`・`transformer`・`vae`・`processor` を持つディレクトリへのパスでも受け付けます。

## 幅とメモリ

チェックポイントは bfloat16 で、transformer だけで 14GB あります。そのため読み込みながら量子化します。既定は 8 ビット、`-q4` で 4 ビットです。

| 幅 | transformer | velocity 誤差 (8 ブロック) |
|---|---|---|
| `-f32` | 28 GB | — |
| 既定 | 7.0 GB | 9.1% |
| `-q4` | 3.5 GB | 13.5% |

誤差は同じモデルの float 版との比較で、深さに対して掛け算では増えません。1 ブロックで 8 ビットが 7.2%、4 ビットが 12.8%、8 ブロックで 9.1% と 13.5% です。どちらもプロンプトどおりの絵になり、4 ビットは細部の解釈が変わります。

70 億パラメータの量子化には数分かかり、結果は毎回同じなので、チェックポイントの隣に `tensai-q8.cache` (または `tensai-q4.cache`) として書き、次回は mmap で読みます。初回は遅く、2 回目からは速くなります。プロンプトエンコーダの起動が 5 分 23 秒から 7 秒、transformer が 7 分 7 秒から 2 秒です。キャッシュは隣の `.safetensors` の名前・サイズ・更新時刻で紐付けてあるので、再ダウンロードすれば作り直します。

## サイズと時間

`-size` はピクセル単位で、32 の倍数に切り下げます (デコーダが 16 ピクセル単位で、チェックポイントの配置がそれを 2 つずつ束ねるため)。16 コアのノート PC、AVX2 ビルドでの実測です。

| サイズ | latent トークン | 1 ステップ | 20 ステップ |
|---|---|---|---|
| 256x256 | 256 | 11 秒 | 3 分 46 秒 |
| 512x512 | 1024 | 50 秒 | 約 17 分 |

トークン数の増加より 1 ステップの伸びが大きいのは、attention が列長の二乗だからです。GPU 経路はありません。WebGPU バックエンドはこの機械で行列積を 6〜7 倍速く回しますが、denoising transformer の rotary embedding は位置軸が 3 本あり隣接ペアを回す形で、GPU 側のカーネルが持つどちらの規約とも違います。

## フラグ

```
tensai image [flags] <prompt>

  -model string   キャッシュ下の名前、またはパス (既定 "Qwen-Image-2.1")
  -o string       書き出し先 (既定 "out.png")
  -size int       幅と高さ、ピクセル (既定 256)
  -steps int      デノイジングのステップ数 (既定 20)
  -seed int       ノイズのシード (既定 1)
  -q4             8 ビットではなく 4 ビットに量子化する
  -f32            重みを float のまま持つ。およそ 42GB 必要
  -negative str   避けたいもの。-cfg を 1 より大きくする必要がある
  -cfg float      どれだけ避けるか (既定 1、off)
  -q              エラー以外を出さない
```

## ガイダンス

`-cfg` を 1 より大きくすると classifier-free guidance が有効になります。各ステップで
transformer に 2 回聞き (プロンプトが求めるものと `-negative` が求めるもの)、その差を
前者の側に伸ばします。色が分離し質感が出て絵が締まりますが、1 ステップのコストが 2 倍に
なるため既定では off です。

```bash
tensai image -cfg 4 -negative "blurry, low quality, watermark" \
  "a calico cat asleep on a stack of books"
```

256x256 の 20 ステップが 3 分 46 秒から 6 分 29 秒になります。2 つのプロンプトは
エンコーダの 1 回の読み込みで両方処理するので、そちらの追加コストはありません。

## まだ無いもの

text-to-image だけです。チェックポイントは画像編集もできますが、そちらは未実装です。プロンプトエンコーダは言語側だけを走らせるので vision tower は使いません。参照実装がステップ間で使い回す key-value キャッシュ (プロンプトは timestep 0 から変調するので全ステップで変わりません) も作っていません。
