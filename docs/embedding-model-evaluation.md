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
