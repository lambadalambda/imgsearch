package parakeetonnx

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os/exec"
)

const SampleRate = 16000

func ExtractPCM16kMono(ctx context.Context, videoPath string) ([]float32, error) {
	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-i", videoPath,
		"-vn",
		"-ac", "1",
		"-ar", fmt.Sprintf("%d", SampleRate),
		"-f", "f32le",
		"pipe:1",
	)
	output, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("extract audio with ffmpeg: %w: %s", err, string(ee.Stderr))
		}
		return nil, fmt.Errorf("extract audio with ffmpeg: %w", err)
	}
	if len(output) == 0 {
		return nil, fmt.Errorf("ffmpeg returned no audio samples")
	}
	if len(output)%4 != 0 {
		return nil, fmt.Errorf("invalid f32le output length %d", len(output))
	}

	samples := make([]float32, len(output)/4)
	if err := binary.Read(bytes.NewReader(output), binary.LittleEndian, samples); err != nil {
		return nil, fmt.Errorf("decode f32le audio: %w", err)
	}
	for i, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return nil, fmt.Errorf("invalid sample at %d", i)
		}
	}
	return samples, nil
}
