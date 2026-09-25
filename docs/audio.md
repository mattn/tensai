# Audio Understanding

`tensai audio` answers a question about a sound file with [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct): transcribe speech, say what a noise is, describe music. It is the same pure Go as the rest.

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

The question is optional; without one the model is asked what it hears.

## What runs

Qwen2-Audio is a Qwen2 language model with Whisper's encoder in front of it.

1. **The front end.** The samples become Whisper's log-mel spectrogram: 128 bands, a 25ms window every 10ms, padded to thirty seconds. It matches transformers' `WhisperFeatureExtractor` to 1e-4.
2. **The encoder.** Whisper-large-v3's 32 layers, then a pooling that halves the sequence again and a projection into the language model's embedding space. Each 40ms of audio becomes one vector. It agrees with transformers to a relative error of 8e-6, checked with the tower cut to two layers, since the wiring is the same at any depth. The reference pads to thirty seconds and masks the padding out; tensai runs the audio's own positions only, which gives the same vectors for less work on a short clip.
3. **The language model.** Those vectors take the place of the prompt's `<|AUDIO|>` placeholder, one position each, and the model answers the way `tensai run` would. It loads through the same code as every other checkpoint, so `-q8`, `-q4`, `-temp`, `-system` and the rest behave as they do there.

The encoder's weights stay float32 (2.5GB) and it is released before the language model loads, so the two are never in memory together. The language model at eight bits is about 8GB; the whole run peaks near 14.6GB, the same as the language model alone.

## Input

`tensai audio` reads WAV: 8, 16, 24 or 32-bit PCM, or 32 or 64-bit float, any number of channels (averaged) and any rate (resampled to 16kHz). Anything else wants converting first:

```bash
ffmpeg -i clip.mp3 clip.wav
```

The model hears thirty seconds at most; a longer file is cut there, and tensai says so.

## Getting the checkpoint

```bash
tensai audio clip.wav "What is this sound?"
```

The default `-model` is `Qwen/Qwen2-Audio-7B-Instruct`, about 16GB, downloaded on first use into `~/.cache/tensai/Qwen/Qwen2-Audio-7B-Instruct`. `tensai models` lists it, and `tensai run -model Qwen/Qwen2-Audio-7B-Instruct` runs its language model on text alone.

## Serving

`tensai serve -model Qwen/Qwen2-Audio-7B-Instruct` takes audio in `/v1/chat/completions` the way OpenAI's API sends it: a message's content as a list of parts, a clip as an `input_audio` part holding base64 WAV.

```json
{"messages": [{"role": "user", "content": [
  {"type": "input_audio", "input_audio": {"data": "UklGR...", "format": "wav"}},
  {"type": "text", "text": "What's that sound?"}]}]}
```

A conversation can carry several clips, numbered in order the way the model's chat template numbers them, and a client that resends the whole history with each turn pays for its audio once: an encoded clip is remembered by its content, so the prompt cache reuses everything up to the new question (a follow-up about a four-second clip took 4.4s against 23.8s for the first). The encoder loads for a request with a clip it has not seen and is released after it, rather than staying resident. Other formats are refused, as is audio sent to a model without an encoder.

## Flags

```
tensai audio [flags] <file.wav> [question]
```

`tensai audio` takes the flags `tensai run` does (`-model`, `-q8`/`-q4`/`-f32`, `-temp`, `-topp`, `-seed`, `-repeat`, `-system`, `-n`, `-v` and the rest). The system prompt defaults to Qwen2-Audio's own, "You are a helpful assistant.", rather than the Qwen identity `run` gives. `-draft` and gguf checkpoints are not supported, since neither has the audio encoder.
