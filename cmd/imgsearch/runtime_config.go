package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
)

type runtimeConfig struct {
	DataDir string
	Addr    string
	APIKey  string

	ModeFlag              string
	Mode                  runtimeMode
	ResolvedAPIKey        string
	UsingDefaultAPIKey    bool
	AnnotationModeWarning string
	LoadAnnotator         bool

	WorkerBatchSize int

	LlamaNativeModelPath              string
	LlamaNativeMMProjPath             string
	LlamaNativeDimensions             int
	LlamaNativeGPULayers              int
	LlamaNativeUseGPU                 bool
	LlamaNativeContextSize            int
	LlamaNativeBatchSize              int
	LlamaNativeMaxSequences           int
	LlamaNativeThreads                int
	LlamaNativeFlashAttnType          int
	LlamaNativeCacheTypeK             int
	LlamaNativeCacheTypeV             int
	LlamaNativeImageMaxSide           int
	LlamaNativeImageMaxTokens         int
	ResolvedLlamaNativeImageMaxSide   int
	ResolvedLlamaNativeImageMaxTokens int
	LlamaNativeAnnotationTemperature  float64
	LlamaNativeAnnotationSeed         int64
	LlamaNativeQueryInstruction       string
	LlamaNativePassageInstruction     string

	ParakeetONNXBundleDir   string
	ParakeetONNXRuntimeLib  string
	ParakeetONNXCoreML      bool
	EnableAnnotations       bool
	AnnotatorModelPath      string
	AnnotatorMMProjPath     string
	AnnotatorVariant        string
	AnnotatorGPULayers      int
	AnnotatorUseGPU         bool
	AnnotatorContextSize    int
	AnnotatorBatchSize      int
	AnnotatorThreads        int
	AnnotatorFlashAttnType  int
	AnnotatorCacheTypeK     int
	AnnotatorCacheTypeV     int
	AnnotatorImageMaxSide   int
	AnnotatorImageMaxTokens int

	VectorBackend    string
	SQLiteVectorPath string
}

func defaultRuntimeConfig(getenv func(string) string) runtimeConfig {
	parakeetBundleDefault := strings.TrimSpace(getenv("PARAKEET_ONNX_BUNDLE_DIR"))
	if parakeetBundleDefault == "" {
		parakeetBundleDefault = defaultParakeetOnnxBundleDir
	}

	return runtimeConfig{
		DataDir: "./data",
		Addr:    "127.0.0.1:8080",
		APIKey:  strings.TrimSpace(getenv("IMGSEARCH_API_KEY")),

		ModeFlag:        string(runtimeModeAll),
		WorkerBatchSize: 1,

		LlamaNativeModelPath:             defaultLlamaNativeModelPath,
		LlamaNativeMMProjPath:            defaultLlamaNativeMMProjPath,
		LlamaNativeDimensions:            defaultLlamaNativeDimensions,
		LlamaNativeGPULayers:             99,
		LlamaNativeUseGPU:                true,
		LlamaNativeContextSize:           defaultLlamaNativeEmbedderContextSize,
		LlamaNativeBatchSize:             512,
		LlamaNativeMaxSequences:          1,
		LlamaNativeThreads:               0,
		LlamaNativeFlashAttnType:         defaultLlamaNativeFlashAttnType,
		LlamaNativeCacheTypeK:            defaultLlamaNativeCacheTypeK,
		LlamaNativeCacheTypeV:            defaultLlamaNativeCacheTypeV,
		LlamaNativeImageMaxSide:          384,
		LlamaNativeImageMaxTokens:        0,
		LlamaNativeAnnotationTemperature: defaultLlamaNativeAnnotationTemperature,
		LlamaNativeAnnotationSeed:        defaultLlamaNativeAnnotationSeed,
		LlamaNativeQueryInstruction:      "Retrieve images or text relevant to the user's query.",
		LlamaNativePassageInstruction:    "Represent this image or text for retrieval.",

		ParakeetONNXBundleDir:   parakeetBundleDefault,
		ParakeetONNXRuntimeLib:  strings.TrimSpace(getenv("PARAKEET_ONNXRUNTIME_LIB")),
		ParakeetONNXCoreML:      getenv("PARAKEET_ONNX_COREML") == "1",
		EnableAnnotations:       true,
		AnnotatorVariant:        defaultLlamaNativeAnnotatorVariant,
		AnnotatorGPULayers:      99,
		AnnotatorUseGPU:         true,
		AnnotatorContextSize:    8192,
		AnnotatorBatchSize:      512,
		AnnotatorThreads:        0,
		AnnotatorFlashAttnType:  defaultLlamaNativeFlashAttnType,
		AnnotatorCacheTypeK:     defaultLlamaNativeCacheTypeK,
		AnnotatorCacheTypeV:     defaultLlamaNativeCacheTypeV,
		AnnotatorImageMaxSide:   1024,
		AnnotatorImageMaxTokens: 0,

		VectorBackend: vectorBackendAuto,
	}
}

func registerRuntimeFlags(fs *flag.FlagSet, cfg *runtimeConfig) {
	fs.StringVar(&cfg.DataDir, "data-dir", cfg.DataDir, "data directory")
	fs.StringVar(&cfg.Addr, "addr", cfg.Addr, "http listen address")
	fs.StringVar(&cfg.APIKey, "api-key", cfg.APIKey, "API key required for /api/* requests (falls back to built-in development default when unset)")
	fs.StringVar(&cfg.ModeFlag, "mode", cfg.ModeFlag, "process mode: all, api, or worker")
	fs.IntVar(&cfg.WorkerBatchSize, "worker-batch-size", cfg.WorkerBatchSize, "number of jobs to claim per batch (1 disables batching)")
	fs.StringVar(&cfg.LlamaNativeModelPath, "llama-native-model-path", cfg.LlamaNativeModelPath, "path to the llama.cpp GGUF embedding model")
	fs.StringVar(&cfg.LlamaNativeMMProjPath, "llama-native-mmproj-path", cfg.LlamaNativeMMProjPath, "path to the llama.cpp GGUF mmproj model")
	fs.IntVar(&cfg.LlamaNativeDimensions, "llama-native-dimensions", cfg.LlamaNativeDimensions, "embedding dimensions for llama-cpp-native model metadata")
	fs.IntVar(&cfg.LlamaNativeGPULayers, "llama-native-gpu-layers", cfg.LlamaNativeGPULayers, "number of layers to offload for llama-cpp-native")
	fs.BoolVar(&cfg.LlamaNativeUseGPU, "llama-native-use-gpu", cfg.LlamaNativeUseGPU, "whether llama-cpp-native should use GPU for mtmd/mmproj")
	fs.IntVar(&cfg.LlamaNativeContextSize, "llama-native-context-size", cfg.LlamaNativeContextSize, "context size for llama-cpp-native runtime")
	fs.IntVar(&cfg.LlamaNativeBatchSize, "llama-native-batch-size", cfg.LlamaNativeBatchSize, "batch size for llama-cpp-native runtime")
	fs.IntVar(&cfg.LlamaNativeMaxSequences, "llama-native-max-sequences", cfg.LlamaNativeMaxSequences, "maximum independent sequences for llama-cpp-native embedding batches")
	fs.IntVar(&cfg.LlamaNativeThreads, "llama-native-threads", cfg.LlamaNativeThreads, "thread count for llama-cpp-native runtime (0 uses backend default)")
	fs.IntVar(&cfg.LlamaNativeFlashAttnType, "llama-native-flash-attn", cfg.LlamaNativeFlashAttnType, "flash attention type for llama-cpp-native (-1=auto, 0=off, 1=on)")
	fs.IntVar(&cfg.LlamaNativeCacheTypeK, "llama-native-cache-type-k", cfg.LlamaNativeCacheTypeK, "K cache type for llama-cpp-native (-1=default, 0=f32, 1=f16, 8=q8_0)")
	fs.IntVar(&cfg.LlamaNativeCacheTypeV, "llama-native-cache-type-v", cfg.LlamaNativeCacheTypeV, "V cache type for llama-cpp-native (-1=default, 0=f32, 1=f16, 8=q8_0)")
	fs.IntVar(&cfg.LlamaNativeImageMaxSide, "llama-native-image-max-side", cfg.LlamaNativeImageMaxSide, "maximum image side length used before llama-cpp-native embedding")
	fs.IntVar(&cfg.LlamaNativeImageMaxTokens, "llama-native-image-max-tokens", cfg.LlamaNativeImageMaxTokens, "optional maximum image tokens override for llama-cpp-native mtmd preprocessing (0 uses model default)")
	fs.Float64Var(&cfg.LlamaNativeAnnotationTemperature, "llama-native-annotation-temperature", cfg.LlamaNativeAnnotationTemperature, "sampling temperature for llama-cpp-native image/video annotation generation (0 for deterministic greedy decode)")
	fs.Int64Var(&cfg.LlamaNativeAnnotationSeed, "llama-native-annotation-seed", cfg.LlamaNativeAnnotationSeed, "RNG seed for llama-cpp-native image/video annotation generation (-1 uses random seed per request)")
	fs.StringVar(&cfg.LlamaNativeQueryInstruction, "llama-native-query-instruction", cfg.LlamaNativeQueryInstruction, "instruction used for llama-cpp-native text query embeddings")
	fs.StringVar(&cfg.LlamaNativePassageInstruction, "llama-native-passage-instruction", cfg.LlamaNativePassageInstruction, "instruction used for llama-cpp-native image/document embeddings")
	fs.StringVar(&cfg.ParakeetONNXBundleDir, "parakeet-onnx-bundle-dir", cfg.ParakeetONNXBundleDir, "directory containing the Parakeet ONNX bundle for video transcription")
	fs.StringVar(&cfg.ParakeetONNXRuntimeLib, "parakeet-onnxruntime-lib", cfg.ParakeetONNXRuntimeLib, "path to the ONNX Runtime shared library used for Parakeet video transcription")
	fs.BoolVar(&cfg.ParakeetONNXCoreML, "parakeet-onnx-coreml", cfg.ParakeetONNXCoreML, "whether to request the CoreML execution provider for Parakeet ONNX video transcription")
	fs.BoolVar(&cfg.EnableAnnotations, "enable-annotations", cfg.EnableAnnotations, "whether to load and run image annotation models")
	fs.StringVar(&cfg.AnnotatorModelPath, "llama-native-annotator-model-path", cfg.AnnotatorModelPath, "optional path to a separate llama.cpp GGUF vision model used only for image descriptions/tags")
	fs.StringVar(&cfg.AnnotatorMMProjPath, "llama-native-annotator-mmproj-path", cfg.AnnotatorMMProjPath, "optional path to a separate llama.cpp GGUF mmproj used only for image descriptions/tags")
	fs.StringVar(&cfg.AnnotatorVariant, "llama-native-annotator-variant", cfg.AnnotatorVariant, "default separate llama.cpp annotator model variant when explicit annotator paths are not set: e4b or 26b")
	fs.IntVar(&cfg.AnnotatorGPULayers, "llama-native-annotator-gpu-layers", cfg.AnnotatorGPULayers, "number of layers to offload for the separate llama-cpp-native annotator")
	fs.BoolVar(&cfg.AnnotatorUseGPU, "llama-native-annotator-use-gpu", cfg.AnnotatorUseGPU, "whether the separate llama-cpp-native annotator should use GPU for mtmd/mmproj")
	fs.IntVar(&cfg.AnnotatorContextSize, "llama-native-annotator-context-size", cfg.AnnotatorContextSize, "context size for the separate llama-cpp-native annotator")
	fs.IntVar(&cfg.AnnotatorBatchSize, "llama-native-annotator-batch-size", cfg.AnnotatorBatchSize, "batch size for the separate llama-cpp-native annotator")
	fs.IntVar(&cfg.AnnotatorThreads, "llama-native-annotator-threads", cfg.AnnotatorThreads, "thread count for the separate llama-cpp-native annotator (0 uses backend default)")
	fs.IntVar(&cfg.AnnotatorFlashAttnType, "llama-native-annotator-flash-attn", cfg.AnnotatorFlashAttnType, "flash attention type for the separate annotator (-1=auto, 0=off, 1=on)")
	fs.IntVar(&cfg.AnnotatorCacheTypeK, "llama-native-annotator-cache-type-k", cfg.AnnotatorCacheTypeK, "K cache type for the separate annotator (-1=default, 0=f32, 1=f16, 8=q8_0)")
	fs.IntVar(&cfg.AnnotatorCacheTypeV, "llama-native-annotator-cache-type-v", cfg.AnnotatorCacheTypeV, "V cache type for the separate annotator (-1=default, 0=f32, 1=f16, 8=q8_0)")
	fs.IntVar(&cfg.AnnotatorImageMaxSide, "llama-native-annotator-image-max-side", cfg.AnnotatorImageMaxSide, "maximum image side length used before separate llama-cpp-native annotation generation")
	fs.IntVar(&cfg.AnnotatorImageMaxTokens, "llama-native-annotator-image-max-tokens", cfg.AnnotatorImageMaxTokens, "optional maximum image tokens override for the separate llama-cpp-native annotator (0 uses model default)")
	fs.StringVar(&cfg.VectorBackend, "vector-backend", cfg.VectorBackend, "vector backend: auto, sqlite-vector, bruteforce")
	fs.StringVar(&cfg.SQLiteVectorPath, "sqlite-vector-path", cfg.SQLiteVectorPath, "path to sqlite-vector extension binary (optional: defaults to SQLITE_VECTOR_PATH or tools/sqlite-vector/vector)")
}

func (cfg *runtimeConfig) Resolve() error {
	mode, err := resolveRuntimeMode(cfg.ModeFlag)
	if err != nil {
		return fmt.Errorf("configure mode: %w", err)
	}
	cfg.Mode = mode
	cfg.ResolvedAPIKey, cfg.UsingDefaultAPIKey = resolveAPIKey(cfg.APIKey)
	if cfg.Mode.startsHTTP() {
		if err := validateHTTPExposure(cfg.Addr, cfg.ResolvedAPIKey, cfg.UsingDefaultAPIKey); err != nil {
			return fmt.Errorf("configure api security: %w", err)
		}
	}
	if cfg.LlamaNativeDimensions <= 0 {
		return fmt.Errorf("configure embedder dimensions: llama-cpp-native dimensions must be positive")
	}
	cfg.ResolvedLlamaNativeImageMaxSide, cfg.ResolvedLlamaNativeImageMaxTokens, err = resolveLlamaCPPNativeImageLimits(cfg.LlamaNativeImageMaxSide, cfg.LlamaNativeImageMaxTokens)
	if err != nil {
		return fmt.Errorf("configure llama-cpp-native options: %w", err)
	}
	cfg.LoadAnnotator = shouldLoadAnnotator(cfg.Mode, cfg.EnableAnnotations)
	cfg.AnnotationModeWarning = annotationModeWarning(cfg.Mode, cfg.EnableAnnotations, cfg.AnnotatorModelPath, cfg.AnnotatorMMProjPath)
	return nil
}

func parseRuntimeConfig(args []string, getenv func(string) string) (runtimeConfig, error) {
	cfg := defaultRuntimeConfig(getenv)
	fs := flag.NewFlagSet("imgsearch", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	registerRuntimeFlags(fs, &cfg)
	if err := fs.Parse(args); err != nil {
		return runtimeConfig{}, err
	}
	if err := cfg.Resolve(); err != nil {
		return runtimeConfig{}, err
	}
	return cfg, nil
}
