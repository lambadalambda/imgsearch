package parakeetonnx

import (
	"context"
	"math"
	"path/filepath"
	"runtime"
	"testing"
)

func TestExtractPCM16kMonoFromFixtureVideo(t *testing.T) {
	path := fixtureVideoPath(t)
	samples, err := ExtractPCM16kMono(context.Background(), path)
	if err != nil {
		t.Fatalf("extract audio: %v", err)
	}
	if len(samples) < SampleRate {
		t.Fatalf("expected at least 1s of audio, got %d samples", len(samples))
	}
	for i, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatalf("invalid sample at %d", i)
		}
	}
}

func TestBuildFeaturesFromFixtureVideo(t *testing.T) {
	path := fixtureVideoPath(t)
	samples, err := ExtractPCM16kMono(context.Background(), path)
	if err != nil {
		t.Fatalf("extract audio: %v", err)
	}
	features, err := BuildFeatures(samples)
	if err != nil {
		t.Fatalf("build features: %v", err)
	}
	if len(features.Values) != FeatureBins {
		t.Fatalf("expected %d feature bins, got %d", FeatureBins, len(features.Values))
	}
	if features.Length <= 0 {
		t.Fatalf("expected positive feature length, got %d", features.Length)
	}
	if len(features.Values[0]) < int(features.Length) {
		t.Fatalf("expected feature frames >= length, got frames=%d length=%d", len(features.Values[0]), features.Length)
	}
	for mel := range features.Values {
		for frame, value := range features.Values[mel] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("invalid feature value at mel=%d frame=%d", mel, frame)
			}
		}
	}
}

func fixtureVideoPath(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..", ".."))
	return filepath.Join(root, "fixtures", "videos", "Tis better to remain silent and be thought a fool [EdDQOVpIpQA].mp4")
}
