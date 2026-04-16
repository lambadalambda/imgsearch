package parakeetonnx

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestTranscribeFixtureVideo(t *testing.T) {
	ortLib := os.Getenv("PARAKEET_SPIKE_ONNXRUNTIME_LIB")
	bundleDir := os.Getenv("PARAKEET_SPIKE_BUNDLE_DIR")
	if ortLib == "" || bundleDir == "" {
		t.Skip("set PARAKEET_SPIKE_ONNXRUNTIME_LIB and PARAKEET_SPIKE_BUNDLE_DIR")
	}

	transcript, err := TranscribeVideo(context.Background(), RecognizerConfig{
		ONNXRuntimeLib: ortLib,
		BundleDir:      bundleDir,
		UseCoreML:      os.Getenv("PARAKEET_SPIKE_COREML") == "1",
	}, fixtureVideoPath(t))
	if err != nil {
		t.Fatalf("transcribe video: %v", err)
	}

	normalized := normalizeText(transcript.Text)
	for _, want := range []string{"better", "remain", "silent", "doubt"} {
		if !strings.Contains(normalized, want) {
			t.Fatalf("expected transcript to contain %q, got %q", want, transcript.Text)
		}
	}
	t.Logf("transcript=%q", transcript.Text)
}

func normalizeText(s string) string {
	s = strings.ToLower(s)
	replacer := strings.NewReplacer(
		",", " ",
		".", " ",
		"!", " ",
		"?", " ",
		"'", " ",
		"\"", " ",
	)
	s = replacer.Replace(s)
	return strings.Join(strings.Fields(s), " ")
}
