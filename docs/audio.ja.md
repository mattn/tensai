# 音声理解

`tensai audio` は [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) で音声ファイルについての質問に答えます。話し言葉の書き起こし、物音の正体、音楽の説明などができます。ほかと同じく純 Go です。

```
$ tensai audio speech.wav "Transcribe the speech exactly."
audio: 5.9s as 146 tokens in 27.43s
loaded qwen2 (32 layers, hidden 4096) as int8 in 1m10s
prompt: 179 tokens
prefill: 29.3s
The original content of this audio is: 'Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'

$ tensai audio glass.wav "これは何の音ですか？日本語で答えてください。"
これは、ガラスが割れた音です。
```

質問は省略できます。省略すると、何が聞こえるかを尋ねます。

## 何が動くか

Qwen2-Audio は、Qwen2 の言語モデルの前に Whisper のエンコーダを置いたものです。

1. **前処理。** サンプルを Whisper の log-mel スペクトログラムにします。128 バンド、25ms の窓を 10ms ごとに取り、30 秒に揃えます。transformers の `WhisperFeatureExtractor` と 1e-4 で一致します。
2. **エンコーダ。** Whisper-large-v3 の 32 層のあと、系列をさらに半分にするプーリングと、言語モデルの埋め込み空間への射影が続きます。音声 40ms が 1 本のベクトルになります。層数を 2 に絞った照合で transformers と相対誤差 8e-6 で一致します (配線は層数によらず同じです)。参照実装は 30 秒までパディングしてマスクで除外しますが、tensai は音声そのものの位置だけを計算します。結果は同じで、短いクリップでは計算がずっと少なく済みます。
3. **言語モデル。** 得たベクトルがプロンプトの `<|AUDIO|>` の位置に 1 つずつ入り、あとは `tensai run` と同じように答えます。読み込みはほかのチェックポイントと同じコードを通るので、`-q8`、`-q4`、`-temp`、`-system` などもそのまま効きます。

エンコーダの重みは float32 (2.5GB) のままで、言語モデルを読む前に解放します。両者が同時にメモリに載ることはありません。言語モデルは 8 ビットで約 8GB、実行全体のピークは約 14.6GB で、言語モデル単体と変わりません。

## 入力

読めるのは WAV です。8、16、24、32 ビットの PCM と 32、64 ビットの浮動小数点に対応し、チャンネル数は問いません (平均して 1 チャンネルにします)。サンプルレートも問いません (16kHz にリサンプルします)。それ以外は先に変換してください。

```bash
ffmpeg -i clip.mp3 clip.wav
```

モデルが聞けるのは最大 30 秒です。それより長いファイルはそこで切り、その旨を表示します。

## チェックポイントの取得

```bash
tensai audio clip.wav "What is this sound?"
```

`-model` の既定は `Qwen/Qwen2-Audio-7B-Instruct` で、約 16GB あります。初回に `~/.cache/tensai/Qwen/Qwen2-Audio-7B-Instruct` へダウンロードします。`tensai models` に表示され、`tensai run -model Qwen/Qwen2-Audio-7B-Instruct` とすれば言語モデルをテキストだけで動かせます。

## サーバ

`tensai serve -model Qwen/Qwen2-Audio-7B-Instruct` とすると、`/v1/chat/completions` が OpenAI の API と同じ形で音声を受けます。メッセージの content を部品のリストにし、音声は base64 の WAV を入れた `input_audio` 部品で送ります。

```json
{"messages": [{"role": "user", "content": [
  {"type": "input_audio", "input_audio": {"data": "UklGR...", "format": "wav"}},
  {"type": "text", "text": "What's that sound?"}]}]}
```

1 つの会話に複数の音声を入れられます。モデルのチャットテンプレートと同じ順に番号を振ります。毎ターン履歴をまるごと送り直すクライアントでも、音声の計算は 1 回で済みます。エンコード済みの音声を中身で覚えておくので、プロンプトキャッシュが新しい質問の直前まで再利用されます (4 秒の音声への追加の質問が、初回の 23.8 秒に対して 4.4 秒でした)。エンコーダは未知の音声を含むリクエストのときだけ読み込み、終わったら解放します。常駐はさせません。WAV 以外の形式や、エンコーダを持たないモデルへの音声は拒否します。

## フラグ

```
tensai audio [flags] <file.wav> [question]
```

`tensai run` と同じフラグを受けます (`-model`、`-q8`/`-q4`/`-f32`、`-temp`、`-topp`、`-seed`、`-repeat`、`-system`、`-n`、`-v` など)。システムプロンプトの既定は、`run` が使う Qwen の自己紹介ではなく、Qwen2-Audio 自身の "You are a helpful assistant." です。`-draft` と gguf のチェックポイントには対応しません。どちらも音声エンコーダを持たないためです。
