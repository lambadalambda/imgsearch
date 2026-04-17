package parakeetonnx

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os/exec"
	"strings"

	"imgsearch/internal/transcribe"
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
			stderr := strings.TrimSpace(string(ee.Stderr))
			if isNoAudioFFmpegError(stderr) {
				return nil, fmt.Errorf("%w: %s", transcribe.ErrNoAudio, stderr)
			}
			return nil, fmt.Errorf("extract audio with ffmpeg: %w: %s", err, stderr)
		}
		return nil, fmt.Errorf("extract audio with ffmpeg: %w", err)
	}
	if len(output) == 0 {
		return nil, transcribe.ErrNoAudio
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

func isNoAudioFFmpegError(stderr string) bool {
	msg := strings.ToLower(stderr)
	return strings.Contains(msg, "no audio") ||
		strings.Contains(msg, "does not contain any stream") ||
		strings.Contains(msg, "matches no streams") ||
		strings.Contains(msg, "contains no stream")
}
