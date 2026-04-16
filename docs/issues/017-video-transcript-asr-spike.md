# 017: Video Transcript Search via Integrated Local ASR

## Status: Completed (Configuration-Gated)

## Goal

Add transcript-backed video search while keeping the system relatively contained inside the existing application, similar in spirit to the current llama.cpp-native embedder integration.

## Spike Summary

An ONNX-based Parakeet v3 spike was attempted using:

- `github.com/yalue/onnxruntime_go`
- `https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx`

### What worked

- `onnxruntime_go` loads successfully on macOS arm64 using the module's bundled `onnxruntime_arm64.dylib`.
- Both Parakeet ONNX sessions load successfully on this machine:
  - `encoder-model.int8.onnx`
  - `decoder_joint-model.int8.onnx`
- Session creation also succeeds when the CoreML execution provider is appended.
- A fixture video can now be processed through the first half of the pipeline:
  - extract mono 16 kHz PCM with `ffmpeg`
  - generate 128-bin NeMo-style log-mel features in Go
  - run those features through the Parakeet encoder successfully
- The spike now produces actual transcript text for a real fixture video by running a minimal Go port of the TDT greedy decoder loop.

### Encoder milestone

The repo now contains an experimental package:

- `internal/transcribe/parakeetonnx`

Implemented pieces:

- `ExtractPCM16kMono` - video/audio to mono 16 kHz `float32` PCM via `ffmpeg`
- `BuildFeatures` - Go port of the 128-bin NeMo log-mel preprocessor used by `onnx-asr`
- `RunEncoder` - encoder-only ONNX Runtime inference

Verification on the fixture video:

- fixture: `fixtures/videos/Tis better to remain silent and be thought a fool [EdDQOVpIpQA].mp4`
- encoder output shape: `[1 1024 166]`
- encoded length: `166`

This proves that the integrated Go + ONNX Runtime path is not just able to load the model; it can also drive the encoder numerically with locally generated features from a real video fixture.

### Decoder milestone

The spike now includes a minimal decoder implementation that:

- loads the Parakeet decoder/joint ONNX session
- initializes recurrent decoder state
- runs the greedy TDT token emission loop frame by frame
- decodes token IDs back into text with the provided vocab

Fixture result:

- input video: `fixtures/videos/Tis better to remain silent and be thought a fool [EdDQOVpIpQA].mp4`
- output transcript:

```text
Remember,'tis better to remain silent and be thoughtful than to open your mouth and remove all doubt. Takes one to no one.
```

This is not word-perfect, but it is recognizably correct and contains the core phrase from the fixture title.

### What the model expects

The encoder does **not** accept raw waveform input. It expects precomputed audio features:

- `audio_signal`: `float32[-1, 128, -1]`
- `length`: `int64[-1]`

The decoder/joint model is stateful and expects:

- `encoder_outputs`: `float32[-1, 1024, -1]`
- `targets`: `int32[-1, -1]`
- `target_length`: `int32[-1]`
- `input_states_1`: `float32[2, -1, 640]`
- `input_states_2`: `float32[2, -1, 640]`

Outputs include logits plus updated recurrent state tensors.

## Landed Feature

The app now supports integrated video transcription with the Parakeet ONNX path when configured with:

- `-parakeet-onnx-bundle-dir`
- `-parakeet-onnxruntime-lib`

When enabled:

- newly uploaded videos enqueue a `transcribe_video` job
- existing videos missing transcripts or transcript embeddings are backfilled at startup
- the worker transcribes each video and stores `videos.transcript_text`
- the transcript text is embedded with the active Qwen text embedder and stored in `video_transcript_embeddings`
- text search merges frame-based video hits with transcript-embedding hits
- the Videos tab shows transcript text

## Conclusion

An integrated Go + ONNX Runtime path looks plausible, but the ONNX files alone are not enough.

The missing pieces are the same things `onnx-asr` implements around the model:

- log-mel spectrogram preprocessing for 128-bin features
- segmentation / chunking or VAD for long-form audio
- TDT / transducer greedy decoding
- timestamp extraction and segment assembly

So the real question is not whether ONNX Runtime can load the model. It can. The real question is whether we want to own the ASR runtime glue in Go.

The spike is now past model loading, preprocessing, encoder invocation, and basic decoding. The remaining work is productization:

- cleaner API around model/session lifecycle
- robust long-form chunking / VAD
- timestamped segment assembly
- transcript storage and search integration

## Recommendation

Two credible next paths:

1. **Go-first integrated path**
   - keep `onnxruntime_go`
   - implement or port just enough preprocessor + greedy decoder logic for Parakeet v3
   - most contained operationally, but more engineering work

2. **Small in-process bridge path**
   - use a runtime that already solves this layer (`transcribe-rs`-style approach)
   - still integrated, but introduces Rust instead of pure Go

Do **not** use `onnx-go` for this path. It is too low-level / incomplete for a modern ASR runtime like Parakeet TDT.

## Spike Artifact

The repo contains a temporary probe command:

- `cmd/parakeetspike`

It now supports:

- model signature inspection
- optional audio extraction + feature generation
- optional encoder execution on a local video fixture

This command remains useful for debugging and inspection, but the underlying Parakeet ONNX path is now integrated into the product feature behind explicit runtime configuration.
