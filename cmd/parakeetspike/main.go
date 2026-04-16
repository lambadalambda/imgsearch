package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"imgsearch/internal/transcribe/parakeetonnx"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	var (
		bundleDir = flag.String("bundle-dir", "", "directory containing Parakeet ONNX bundle files")
		ortLib    = flag.String("onnxruntime-lib", "", "path to onnxruntime shared library")
		useCoreML = flag.Bool("coreml", false, "try creating sessions with CoreML execution provider")
		videoPath = flag.String("video", "", "optional fixture video to extract audio/features and run through encoder")
	)
	flag.Parse()

	if *bundleDir == "" {
		log.Fatal("-bundle-dir is required")
	}
	if *ortLib == "" {
		log.Fatal("-onnxruntime-lib is required")
	}

	ort.SetSharedLibraryPath(*ortLib)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("initialize onnxruntime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("destroy onnxruntime: %v", err)
		}
	}()

	fmt.Printf("ONNX Runtime version: %s\n", ort.GetVersion())

	options, err := ort.NewSessionOptions()
	if err != nil {
		log.Fatalf("new session options: %v", err)
	}
	defer options.Destroy()
	if *useCoreML {
		if err := options.AppendExecutionProviderCoreMLV2(map[string]string{}); err != nil {
			log.Fatalf("append CoreML provider: %v", err)
		}
	}

	for _, name := range []string{"encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"} {
		path := filepath.Join(*bundleDir, name)
		if _, err := os.Stat(path); err != nil {
			log.Fatalf("stat %s: %v", path, err)
		}
		fmt.Printf("\n== %s ==\n", name)
		inputs, outputs, err := ort.GetInputOutputInfoWithOptions(path, options)
		if err != nil {
			log.Fatalf("inspect %s: %v", name, err)
		}
		for i, input := range inputs {
			fmt.Printf("input[%d] name=%q type=%v shape=%v\n", i, input.Name, input.DataType, input.Dimensions)
		}
		for i, output := range outputs {
			fmt.Printf("output[%d] name=%q type=%v shape=%v\n", i, output.Name, output.DataType, output.Dimensions)
		}
	}

	if *videoPath == "" {
		return
	}

	samples, err := parakeetonnx.ExtractPCM16kMono(context.Background(), *videoPath)
	if err != nil {
		log.Fatalf("extract audio: %v", err)
	}
	features, err := parakeetonnx.BuildFeatures(samples)
	if err != nil {
		log.Fatalf("build features: %v", err)
	}
	fmt.Printf("\n== features ==\n")
	fmt.Printf("audio_samples=%d feature_bins=%d feature_frames=%d feature_length=%d\n", len(samples), len(features.Values), len(features.Values[0]), features.Length)

	result, err := parakeetonnx.RunEncoder(parakeetonnx.EncoderConfig{
		ONNXRuntimeLib: *ortLib,
		EncoderPath:    filepath.Join(*bundleDir, "encoder-model.int8.onnx"),
		UseCoreML:      *useCoreML,
	}, features)
	if err != nil {
		log.Fatalf("run encoder: %v", err)
	}
	fmt.Printf("\n== encoder run ==\n")
	fmt.Printf("output_shape=%v encoded_length=%d hidden_size=%d\n", result.OutputShape, result.EncodedLength, result.HiddenSize)

	transcript, err := parakeetonnx.TranscribeVideo(context.Background(), parakeetonnx.RecognizerConfig{
		ONNXRuntimeLib: *ortLib,
		BundleDir:      *bundleDir,
		UseCoreML:      *useCoreML,
	}, *videoPath)
	if err != nil {
		log.Fatalf("transcribe video: %v", err)
	}
	fmt.Printf("\n== transcript ==\n")
	fmt.Printf("text=%q\n", transcript.Text)
	fmt.Printf("token_count=%d\n", len(transcript.TokenIDs))
}
