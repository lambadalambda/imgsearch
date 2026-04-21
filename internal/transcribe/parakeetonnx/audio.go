package parakeetonnx

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strconv"
	"strings"

	"imgsearch/internal/transcribe"
)

const SampleRate = 16000

const (
	maxTranscriptionDurationSeconds = 10 * 60
	maxTranscriptionSamples         = SampleRate * maxTranscriptionDurationSeconds
)

var probeDurationSecondsFunc = probeDurationSeconds

func ExtractPCM16kMono(ctx context.Context, videoPath string) ([]float32, error) {
	durationSeconds, err := probeDurationSecondsFunc(ctx, videoPath)
	if err != nil {
		return nil, fmt.Errorf("probe audio duration: %w", err)
	}
	if durationSeconds > maxTranscriptionDurationSeconds {
		return nil, fmt.Errorf(
			"%w: duration %.2fs exceeds max %ds",
			transcribe.ErrAudioTooLong,
			durationSeconds,
			maxTranscriptionDurationSeconds,
		)
	}

	cmd := exec.CommandContext(ctx,
		"ffmpeg",
		"-hide_banner",
		"-loglevel", "error",
		"-i", videoPath,
		"-t", strconv.Itoa(maxTranscriptionDurationSeconds),
		"-vn",
		"-ac", "1",
		"-ar", fmt.Sprintf("%d", SampleRate),
		"-f", "f32le",
		"pipe:1",
	)
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("open ffmpeg stdout pipe: %w", err)
	}

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start ffmpeg: %w", err)
	}

	samples, err := decodePCMFloat32Stream(stdoutPipe, maxTranscriptionSamples)
	if err != nil {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
		if errors.Is(err, transcribe.ErrAudioTooLong) {
			return nil, err
		}
		return nil, fmt.Errorf("decode f32le audio stream: %w", err)
	}

	if err := cmd.Wait(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			stderrText := strings.TrimSpace(stderr.String())
			if isNoAudioFFmpegError(stderrText) {
				return nil, fmt.Errorf("%w: %s", transcribe.ErrNoAudio, stderrText)
			}
			if stderrText != "" {
				return nil, fmt.Errorf("extract audio with ffmpeg: %w: %s", ee, stderrText)
			}
		}
		return nil, fmt.Errorf("extract audio with ffmpeg: %w", err)
	}
	if len(samples) == 0 {
		return nil, transcribe.ErrNoAudio
	}

	return samples, nil
}

func decodePCMFloat32Stream(reader io.Reader, maxSamples int) ([]float32, error) {
	if maxSamples <= 0 {
		return nil, fmt.Errorf("max sample count must be positive")
	}

	const chunkSizeBytes = 32 << 10
	chunk := make([]byte, chunkSizeBytes)
	pending := [4]byte{}
	pendingLen := 0
	totalBytes := 0
	samples := make([]float32, 0)

	appendSample := func(raw []byte) error {
		if len(samples) >= maxSamples {
			return fmt.Errorf(
				"%w: sample count %d exceeds max %d",
				transcribe.ErrAudioTooLong,
				len(samples)+1,
				maxSamples,
			)
		}
		sample := math.Float32frombits(binary.LittleEndian.Uint32(raw))
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return fmt.Errorf("invalid sample at %d", len(samples))
		}
		samples = append(samples, sample)
		return nil
	}

	for {
		n, err := reader.Read(chunk)
		if n > 0 {
			totalBytes += n
			data := chunk[:n]

			if pendingLen > 0 {
				need := 4 - pendingLen
				if len(data) < need {
					copy(pending[pendingLen:], data)
					pendingLen += len(data)
					data = nil
				} else {
					copy(pending[pendingLen:], data[:need])
					if err := appendSample(pending[:]); err != nil {
						return nil, err
					}
					pendingLen = 0
					data = data[need:]
				}
			}

			fullBytes := len(data) - len(data)%4
			for i := 0; i < fullBytes; i += 4 {
				if err := appendSample(data[i : i+4]); err != nil {
					return nil, err
				}
			}

			if rem := len(data) - fullBytes; rem > 0 {
				copy(pending[:], data[fullBytes:])
				pendingLen = rem
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read ffmpeg stdout: %w", err)
		}
	}

	if pendingLen != 0 {
		return nil, fmt.Errorf("invalid f32le output length %d", totalBytes)
	}

	return samples, nil
}

func probeDurationSeconds(ctx context.Context, videoPath string) (float64, error) {
	cmd := exec.CommandContext(ctx,
		"ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath,
	)
	output, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return 0, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(ee.Stderr)))
		}
		return 0, err
	}
	value := strings.TrimSpace(string(output))
	if value == "" {
		return 0, fmt.Errorf("ffprobe returned empty duration")
	}
	durationSeconds, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration %q: %w", value, err)
	}
	if durationSeconds < 0 {
		return 0, fmt.Errorf("invalid negative duration %.2f", durationSeconds)
	}
	return durationSeconds, nil
}

func isNoAudioFFmpegError(stderr string) bool {
	msg := strings.ToLower(stderr)
	return strings.Contains(msg, "no audio") ||
		strings.Contains(msg, "does not contain any stream") ||
		strings.Contains(msg, "matches no streams") ||
		strings.Contains(msg, "contains no stream")
}
