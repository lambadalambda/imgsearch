package main

import (
	"strings"
	"testing"
)

func TestParseRuntimeConfigResolvesModeAndAPIKey(t *testing.T) {
	cfg, err := parseRuntimeConfig([]string{"-mode", "api", "-api-key", "secret-token"}, func(string) string { return "" })
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.Mode != runtimeModeAPI {
		t.Fatalf("mode: got=%q want=%q", cfg.Mode, runtimeModeAPI)
	}
	if cfg.ResolvedAPIKey != "secret-token" || cfg.UsingDefaultAPIKey {
		t.Fatalf("unexpected api key resolution: key=%q default=%t", cfg.ResolvedAPIKey, cfg.UsingDefaultAPIKey)
	}
}

func TestParseRuntimeConfigRejectsExposedDefaultAPIKey(t *testing.T) {
	_, err := parseRuntimeConfig([]string{"-addr", "0.0.0.0:8080"}, func(string) string { return "" })
	if err == nil || !strings.Contains(err.Error(), "configure api security") {
		t.Fatalf("expected api security error, got %v", err)
	}
}

func TestParseRuntimeConfigAllowsWorkerModeWithoutAPIKey(t *testing.T) {
	cfg, err := parseRuntimeConfig([]string{"-mode", "worker", "-addr", "0.0.0.0:8080"}, func(string) string { return "" })
	if err != nil {
		t.Fatalf("worker config should not validate http exposure: %v", err)
	}
	if cfg.Mode != runtimeModeWorker || cfg.Mode.startsHTTP() {
		t.Fatalf("unexpected worker mode: %+v", cfg.Mode)
	}
}

func TestParseRuntimeConfigRejectsInvalidNativeImageLimits(t *testing.T) {
	_, err := parseRuntimeConfig([]string{"-llama-native-image-max-side", "-1"}, func(string) string { return "" })
	if err == nil || !strings.Contains(err.Error(), "configure llama-cpp-native options") {
		t.Fatalf("expected native image limits error, got %v", err)
	}
}

func TestDefaultRuntimeConfigUsesEnvironment(t *testing.T) {
	cfg := defaultRuntimeConfig(func(key string) string {
		switch key {
		case "IMGSEARCH_API_KEY":
			return "env-key"
		case "PARAKEET_ONNX_BUNDLE_DIR":
			return "/bundle"
		case "PARAKEET_ONNXRUNTIME_LIB":
			return "/libonnxruntime.so"
		case "PARAKEET_ONNX_COREML":
			return "1"
		default:
			return ""
		}
	})
	if cfg.APIKey != "env-key" || cfg.ParakeetONNXBundleDir != "/bundle" || cfg.ParakeetONNXRuntimeLib != "/libonnxruntime.so" || !cfg.ParakeetONNXCoreML {
		t.Fatalf("unexpected env defaults: %+v", cfg)
	}
}

func TestParseRuntimeConfigKeepsFlagDefaults(t *testing.T) {
	cfg, err := parseRuntimeConfig(nil, func(string) string { return "" })
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	if cfg.DataDir != "./data" || cfg.Addr != "127.0.0.1:8080" || cfg.Mode != runtimeModeAll {
		t.Fatalf("unexpected basic defaults: %+v", cfg)
	}
	if cfg.LlamaNativeModelPath != llamaNativeSearch2BModelPath || cfg.LlamaNativeMMProjPath != llamaNativeSearch2BMMProjPath {
		t.Fatalf("unexpected native model defaults: %+v", cfg)
	}
	if cfg.AnnotatorVariant != defaultLlamaNativeAnnotatorVariant {
		t.Fatalf("unexpected annotator default: %+v", cfg)
	}
	if cfg.LlamaNativeDimensions != 2048 || defaultLlamaNativeDimensions != 2048 || cfg.LlamaNativeImageMaxSide != 384 || cfg.ResolvedLlamaNativeImageMaxSide != 384 {
		t.Fatalf("unexpected native defaults: %+v", cfg)
	}
	if cfg.VectorBackend != vectorBackendAuto || !cfg.EnableAnnotations {
		t.Fatalf("unexpected backend/annotation defaults: %+v", cfg)
	}
}
