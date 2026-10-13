# Embedding model evaluation

`cmd/retrievalbench` scores a native embedding model on a captioned image set. It reports text-to-image and image-to-text Recall@1/5/10, optionally at Matryoshka-truncated dimensions, plus embedding time per image and per caption.

The fixture integration tests (`mise run llama-cpp-native-test-fixtures`) only check that a model is usable. With 13 images, they cannot rank two models.

## Dataset

We use the Flickr30k 1k test split (Karpathy split): 1,000 images and 5,000 captions. It is for research use. Keep it outside the repository:

```bash
hf download nlphuji/flickr_1k_test_image_text_retrieval --repo-type dataset --local-dir ../imgsearch-bench/flickr1k
(cd ../imgsearch-bench/flickr1k && unzip -q images_flickr_1k_test.zip)
```

## Running

```bash
mise run bench:retrieval -- \
  -manifest ../imgsearch-bench/flickr1k/test_1k_flickr.csv \
  -images-dir ../imgsearch-bench/flickr1k/images_flickr_1k_test \
  -model models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/Qwen3-VL-Embedding-2B-Q6_K.gguf \
  -mmproj models/VesNFF/Qwen3-VL-Embedding-2B-GGUF/mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  -truncate 0,1024,512,256
```

Use `-prompt-format wemm` for WeMM-Embedding models, and `-limit N` for a quick run. Images go through the same native path as the server, with `-image-max-side 384` by default. The results include the effect of our preprocessing.

## Results: WeMM-Embedding-2B vs Qwen3-VL-Embedding (2026-10-13)

Flickr30k 1k test, Apple M2 Max (Metal), `-image-max-side 384`, one sequence. Recall in %, time per item.

| Model | Quant | Dims | T2I@1 | T2I@5 | T2I@10 | I2T@1 | I2T@5 | ms/image | ms/caption |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-Embedding-2B (default) | Q6_K | 2048 | 80.8 | 94.7 | 97.2 | 94.3 | 100.0 | 242 | 60 |
| Qwen3-VL-Embedding-8B | Q4_K_M | 4096 | 81.1 | 95.0 | 97.4 | 94.5 | 99.6 | 799 | 222 |
| WeMM-Embedding-2B | Q6_K | 2048 | **84.2** | **96.6** | **98.1** | **94.7** | 99.6 | **202** | **39** |
| WeMM-Embedding-2B | Q8_0 | 2048 | 84.2 | 96.6 | 98.1 | 94.6 | 99.6 | 191 | 36 |

Matryoshka truncation (T2I@1): WeMM stays at 82.7 at 256 dims, which is still better than either Qwen model at full size. Qwen3-VL-2B falls to 76.2 at 256 dims and the 8B to 75.8.

Findings:

- WeMM-2B is +3.4 points T2I@1 over our default and is faster for images and text. Q6_K loses nothing against Q8_0.
- The text speedup comes mostly from the shorter prompt: WeMM has no system instruction.
- The 8B model is only marginally better than the 2B on this set, at about 4× the cost.
- WeMM's prompt must match the transformers processor output exactly (`<|im_start|>user\n…<|im_end|>\n<embedding>`). The vLLM template without newlines scores 83.7 T2I@1.
- Against the transformers reference, our GGUF text embeddings have cosine 0.9999. Image embeddings are at 0.964–0.979 with the same pixels and token count, independent of quant and backend. See [the mtmd vision gap issue](../meta/issues/investigate-mtmd-vision-reference-gap.md).
- The existing native integration tests (semantic, fixtures, batch-matches-sequential) pass with `-llama-native-prompt-format wemm`.

Caveats: these are captioned photos only. We did not measure video frames, memes, screenshots or text-heavy images on our own library.

### NAS (production hardware)

CT107, AMD Radeon 780M iGPU (Vulkan, `KHR_coopmat`), commit `924c9bb`, reusing the deployed llama.cpp build. The production service was not changed.

| Model | Quant | Images | T2I@1 | I2T@1 | ms/image | ms/caption |
|---|---|---|---|---|---|---|
| WeMM-Embedding-2B | Q4_K_M | 1000 | 84.1 | 95.0 | 421 | 69 |
| WeMM-Embedding-2B | Q4_K_M | first 100 | 96.6 | 100.0 | 430 | 70 |
| Qwen3-VL-Embedding-8B (production) | Q4_K_M | first 100 | 94.8 | 98.0 | 1095 | 325 |

- Q4_K_M loses nothing against Q6_K (84.1 vs 84.2 T2I@1), so it is the better choice for the NAS.
- Against the production 8B, WeMM Q4_K_M is 2.5× faster per image and 4.7× faster per text query. The 8B was timed on 100 images only; its 1k quality is known from the Mac run (81.1).
- Rough re-embed cost for the current library (about 4,900 images plus about 2,300 sampled video frames): about 50 minutes with WeMM Q4_K_M against more than 2 hours with the 8B.

## Converting WeMM-Embedding to GGUF

No official GGUF exists. The vendored converter works with two adjustments:

1. WeMM's `tokenizer.json` adds a normalizer (`\n$` removal) and a post-processor (appends `<embedding>`). Both change the converter's pre-tokenizer hash. Convert from a copy where `normalizer` is `{"type": "NFC"}` and `post_processor` is `null`. Vocab and merges are unchanged, `<embedding>` (id 248077) stays a control token, and the prompt adds it explicitly.
2. Pass `--no-mtp`: the config declares an MTP layer, but the weights do not include it.

```bash
python deps/llama.cpp/convert_hf_to_gguf.py <staged-dir> --no-mtp --outtype f16 --outfile WeMM-Embedding-2B-f16.gguf
python deps/llama.cpp/convert_hf_to_gguf.py <staged-dir> --mmproj --outtype f16 --outfile mmproj-WeMM-Embedding-2B-f16.gguf
llama-quantize WeMM-Embedding-2B-f16.gguf WeMM-Embedding-2B-Q6_K.gguf Q6_K
```
