package parakeetonnx

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestRunEncoderOnFixtureVideo(t *testing.T) {
	ortLib := os.Getenv("PARAKEET_SPIKE_ONNXRUNTIME_LIB")
	bundleDir := os.Getenv("PARAKEET_SPIKE_BUNDLE_DIR")
	if ortLib == "" || bundleDir == "" {
		t.Skip("set PARAKEET_SPIKE_ONNXRUNTIME_LIB and PARAKEET_SPIKE_BUNDLE_DIR")
	}

	samples, err := ExtractPCM16kMono(context.Background(), fixtureVideoPath(t))
	if err != nil {
		t.Fatalf("extract audio: %v", err)
	}
	features, err := BuildFeatures(samples)
	if err != nil {
		t.Fatalf("build features: %v", err)
	}

	result, err := RunEncoder(EncoderConfig{
		ONNXRuntimeLib: ortLib,
		EncoderPath:    filepath.Join(bundleDir, "encoder-model.int8.onnx"),
		UseCoreML:      os.Getenv("PARAKEET_SPIKE_COREML") == "1",
	}, features)
	if err != nil {
		t.Fatalf("run encoder: %v", err)
	}
	if result.HiddenSize != 1024 {
		t.Fatalf("unexpected hidden size: got=%d want=1024", result.HiddenSize)
	}
	if result.EncodedLength <= 0 {
		t.Fatalf("expected positive encoded length, got %d", result.EncodedLength)
	}
	if result.Frames < result.EncodedLength {
		t.Fatalf("expected output frames >= encoded length, got frames=%d encoded_length=%d", result.Frames, result.EncodedLength)
	}
	t.Logf("encoder output shape=%v encoded_length=%d", result.OutputShape, result.EncodedLength)
}
