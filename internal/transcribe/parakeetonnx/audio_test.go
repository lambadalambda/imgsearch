package parakeetonnx

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"imgsearch/internal/transcribe"
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

func TestBuildFeaturesRejectsAudioOverLimit(t *testing.T) {
	samples := make([]float32, maxTranscriptionSamples+1)
	_, err := BuildFeatures(samples)
	if !errors.Is(err, transcribe.ErrAudioTooLong) {
		t.Fatalf("expected ErrAudioTooLong, got %v", err)
	}
}

func TestBuildFeaturesRejectsEmptyAudio(t *testing.T) {
	_, err := BuildFeatures(nil)
	if err == nil {
		t.Fatal("expected error for empty audio")
	}
	if !strings.Contains(err.Error(), "audio samples are empty") {
		t.Fatalf("expected empty-audio message, got %v", err)
	}
}

func TestDecodePCMFloat32StreamReadsSamples(t *testing.T) {
	want := []float32{0.25, -0.5, 1.0}
	encoded := encodeF32LE(t, want)

	got, err := decodePCMFloat32Stream(bytes.NewReader(encoded), len(want))
	if err != nil {
		t.Fatalf("decode stream: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("expected %d samples, got %d", len(want), len(got))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("sample %d: got %f want %f", i, got[i], want[i])
		}
	}
}

func TestDecodePCMFloat32StreamRejectsAudioOverLimit(t *testing.T) {
	encoded := encodeF32LE(t, []float32{0.25, -0.5})

	_, err := decodePCMFloat32Stream(bytes.NewReader(encoded), 1)
	if !errors.Is(err, transcribe.ErrAudioTooLong) {
		t.Fatalf("expected ErrAudioTooLong, got %v", err)
	}
}

func TestDecodePCMFloat32StreamRejectsInvalidLength(t *testing.T) {
	_, err := decodePCMFloat32Stream(bytes.NewReader([]byte{0x01, 0x02, 0x03}), 16)
	if err == nil {
		t.Fatal("expected invalid output length error")
	}
	if !strings.Contains(err.Error(), "invalid f32le output length") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDecodePCMFloat32StreamRejectsNonFiniteSamples(t *testing.T) {
	encoded := encodeF32LE(t, []float32{float32(math.NaN())})

	_, err := decodePCMFloat32Stream(bytes.NewReader(encoded), 16)
	if err == nil {
		t.Fatal("expected invalid sample error")
	}
	if !strings.Contains(err.Error(), "invalid sample") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestDecodePCMFloat32StreamRejectsNonPositiveLimit(t *testing.T) {
	_, err := decodePCMFloat32Stream(bytes.NewReader(nil), 0)
	if err == nil {
		t.Fatal("expected non-positive limit error")
	}
	if !strings.Contains(err.Error(), "max sample count must be positive") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestExtractPCM16kMonoRejectsOverDurationBeforeFFmpeg(t *testing.T) {
	originalProbe := probeDurationSecondsFunc
	t.Cleanup(func() {
		probeDurationSecondsFunc = originalProbe
	})

	probeDurationSecondsFunc = func(context.Context, string) (float64, error) {
		return float64(maxTranscriptionDurationSeconds) + 0.25, nil
	}

	_, err := ExtractPCM16kMono(context.Background(), "nonexistent.mp4")
	if !errors.Is(err, transcribe.ErrAudioTooLong) {
		t.Fatalf("expected ErrAudioTooLong, got %v", err)
	}
}

func TestExtractPCM16kMonoAllowsExactDurationLimit(t *testing.T) {
	originalProbe := probeDurationSecondsFunc
	t.Cleanup(func() {
		probeDurationSecondsFunc = originalProbe
	})

	probeDurationSecondsFunc = func(context.Context, string) (float64, error) {
		return float64(maxTranscriptionDurationSeconds), nil
	}

	_, err := ExtractPCM16kMono(context.Background(), "nonexistent.mp4")
	if err == nil {
		t.Fatal("expected ffmpeg error")
	}
	if errors.Is(err, transcribe.ErrAudioTooLong) {
		t.Fatalf("expected non-limit error at exact duration cap, got %v", err)
	}
}

func TestExtractPCM16kMonoWrapsProbeError(t *testing.T) {
	originalProbe := probeDurationSecondsFunc
	t.Cleanup(func() {
		probeDurationSecondsFunc = originalProbe
	})

	probeSentinel := errors.New("probe boom")
	probeDurationSecondsFunc = func(context.Context, string) (float64, error) {
		return 0, probeSentinel
	}

	_, err := ExtractPCM16kMono(context.Background(), "nonexistent.mp4")
	if !errors.Is(err, probeSentinel) {
		t.Fatalf("expected wrapped probe error, got %v", err)
	}
}

func encodeF32LE(t *testing.T, samples []float32) []byte {
	t.Helper()

	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, samples); err != nil {
		t.Fatalf("encode samples: %v", err)
	}
	return buf.Bytes()
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
