// Command retrievalbench reports text/image Recall@K for a native embedding
// model on a captioned image set (Flickr30k 1k test CSV layout).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"imgsearch/internal/embedder/llamacppnative"
	"imgsearch/internal/retrievalbench"
)

func main() {
	var (
		manifest   = flag.String("manifest", "", "Flickr-style CSV with filename and raw caption columns")
		imagesDir  = flag.String("images-dir", "", "directory containing the manifest images")
		limit      = flag.Int("limit", 0, "evaluate only the first N images (0 = all)")
		truncate   = flag.String("truncate", "0", "comma-separated Matryoshka dimensions to score (0 = full)")
		cfg        llamacppnative.Config
		imageBatch = flag.Int("image-batch", 1, "images per EmbedImages call")
	)
	flag.StringVar(&cfg.ModelPath, "model", "", "text model GGUF")
	flag.StringVar(&cfg.VisionModelPath, "mmproj", "", "vision projector GGUF")
	flag.IntVar(&cfg.Dimensions, "dimensions", 2048, "native embedding dimensions")
	flag.IntVar(&cfg.GPULayers, "gpu-layers", 99, "layers to offload")
	flag.BoolVar(&cfg.UseGPU, "use-gpu", true, "use GPU")
	flag.IntVar(&cfg.ContextSize, "context-size", 8192, "context size")
	flag.IntVar(&cfg.BatchSize, "batch-size", 512, "batch size")
	flag.IntVar(&cfg.MaxSequences, "max-sequences", 1, "parallel sequences")
	flag.IntVar(&cfg.ImageMaxSide, "image-max-side", 384, "image max side before embedding")
	flag.StringVar(&cfg.QueryInstruction, "query-instruction", "", "query instruction (empty = default)")
	flag.StringVar(&cfg.PassageInstruction, "passage-instruction", "", "passage instruction (empty = default)")
	flag.Func("prompt-format", "embedding prompt format: qwen3vl (default) or wemm", func(s string) (err error) {
		cfg.PromptFormat, err = llamacppnative.ParsePromptFormat(s)
		return err
	})
	flag.Parse()
	if *imageBatch < 1 {
		log.Fatal("-image-batch must be at least 1")
	}

	dims, err := parseInts(*truncate)
	if err != nil {
		log.Fatalf("-truncate: %v", err)
	}

	f, err := os.Open(*manifest)
	if err != nil {
		log.Fatalf("open manifest: %v", err)
	}
	items, err := retrievalbench.ParseFlickrCSV(f)
	f.Close()
	if err != nil {
		log.Fatalf("parse manifest: %v", err)
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}

	embedder, err := llamacppnative.New(cfg)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer embedder.Close()
	ctx := context.Background()

	start := time.Now()
	images := make([][]float32, 0, len(items))
	for i := 0; i < len(items); i += *imageBatch {
		end := min(i+*imageBatch, len(items))
		paths := make([]string, 0, end-i)
		for _, item := range items[i:end] {
			paths = append(paths, filepath.Join(*imagesDir, item.Image))
		}
		vecs, err := embedder.EmbedImages(ctx, paths)
		if err != nil {
			log.Fatalf("embed images %v: %v", paths, err)
		}
		images = append(images, vecs...)
	}
	imageTime := time.Since(start)

	start = time.Now()
	var captions [][]float32
	var captionImage []int
	for i, item := range items {
		for _, caption := range item.Captions {
			vec, err := embedder.EmbedText(ctx, caption)
			if err != nil {
				log.Fatalf("embed caption %q: %v", caption, err)
			}
			captions = append(captions, vec)
			captionImage = append(captionImage, i)
		}
	}
	textTime := time.Since(start)

	fmt.Printf("model=%s images=%d captions=%d\n", filepath.Base(cfg.ModelPath), len(images), len(captions))
	fmt.Printf("image embed: %.1f ms/image; text embed: %.1f ms/caption\n",
		ms(imageTime)/float64(len(images)), ms(textTime)/float64(len(captions)))
	ks := []int{1, 5, 10}
	fmt.Println("dims  T2I@1  T2I@5  T2I@10  I2T@1  I2T@5  I2T@10")
	for _, d := range dims {
		r := retrievalbench.Evaluate(truncateAll(images, d), truncateAll(captions, d), captionImage, ks)
		label := strconv.Itoa(d)
		if d == 0 {
			label = "full"
		}
		fmt.Printf("%-5s %5.1f  %5.1f  %6.1f  %5.1f  %5.1f  %6.1f\n", label,
			100*r.TextToImage[1], 100*r.TextToImage[5], 100*r.TextToImage[10],
			100*r.ImageToText[1], 100*r.ImageToText[5], 100*r.ImageToText[10])
	}
}

func truncateAll(vecs [][]float32, dims int) [][]float32 {
	out := make([][]float32, len(vecs))
	for i, v := range vecs {
		out[i] = retrievalbench.TruncateNormalize(v, dims)
	}
	return out
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, nil
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
