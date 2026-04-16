package upload

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

type SampledFrame struct {
	Path        string
	TimestampMS int64
	FrameIndex  int
}

type VideoSample struct {
	DurationMS int64
	Width      int
	Height     int
	Frames     []SampledFrame
}

type VideoSampler interface {
	Sample(ctx context.Context, videoPath string, frameCount int, tmpDir string) (VideoSample, error)
}

type execVideoSampler struct{}

type ffprobeOutput struct {
	Streams []struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"streams"`
	Format struct {
		Duration string `json:"duration"`
	} `json:"format"`
}

func (execVideoSampler) Sample(ctx context.Context, videoPath string, frameCount int, tmpDir string) (VideoSample, error) {
	if frameCount <= 0 {
		frameCount = 10
	}
	probe, err := probeVideo(ctx, videoPath)
	if err != nil {
		return VideoSample{}, err
	}

	timestamps := uniformSampleTimestamps(probe.DurationMS, frameCount)
	out := VideoSample{DurationMS: probe.DurationMS, Width: probe.Width, Height: probe.Height}
	for i, ts := range timestamps {
		framePath := filepath.Join(tmpDir, fmt.Sprintf("frame-%02d.jpg", i))
		cmd := exec.CommandContext(ctx, "ffmpeg",
			"-hide_banner",
			"-loglevel", "error",
			"-y",
			"-ss", formatTimestampSeconds(ts),
			"-i", videoPath,
			"-frames:v", "1",
			"-q:v", "2",
			framePath,
		)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return VideoSample{}, fmt.Errorf("extract video frame %d with ffmpeg: %w: %s", i, err, string(output))
		}
		if _, err := os.Stat(framePath); err != nil {
			return VideoSample{}, fmt.Errorf("stat extracted frame %d: %w", i, err)
		}
		out.Frames = append(out.Frames, SampledFrame{Path: framePath, TimestampMS: ts, FrameIndex: i})
	}
	return out, nil
}

func probeVideo(ctx context.Context, videoPath string) (VideoSample, error) {
	cmd := exec.CommandContext(ctx, "ffprobe",
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height:format=duration",
		"-of", "json",
		videoPath,
	)
	output, err := cmd.Output()
	if err != nil {
		if ee := new(exec.ExitError); errors.As(err, &ee) {
			return VideoSample{}, fmt.Errorf("probe video with ffprobe: %w: %s", err, string(ee.Stderr))
		}
		return VideoSample{}, fmt.Errorf("probe video with ffprobe: %w", err)
	}

	var parsed ffprobeOutput
	if err := json.Unmarshal(output, &parsed); err != nil {
		return VideoSample{}, fmt.Errorf("decode ffprobe output: %w", err)
	}
	if len(parsed.Streams) == 0 || parsed.Streams[0].Width <= 0 || parsed.Streams[0].Height <= 0 {
		return VideoSample{}, fmt.Errorf("ffprobe returned no usable video stream")
	}
	durationSeconds, err := strconv.ParseFloat(parsed.Format.Duration, 64)
	if err != nil {
		return VideoSample{}, fmt.Errorf("parse video duration: %w", err)
	}
	return VideoSample{
		DurationMS: int64(math.Round(durationSeconds * 1000)),
		Width:      parsed.Streams[0].Width,
		Height:     parsed.Streams[0].Height,
	}, nil
}

func uniformSampleTimestamps(durationMS int64, frameCount int) []int64 {
	if frameCount <= 0 {
		return nil
	}
	out := make([]int64, 0, frameCount)
	if durationMS <= 0 {
		for range frameCount {
			out = append(out, 0)
		}
		return out
	}
	for i := 0; i < frameCount; i++ {
		fraction := (float64(i) + 0.5) / float64(frameCount)
		ts := int64(math.Round(float64(durationMS) * fraction))
		if ts >= durationMS {
			ts = durationMS - 1
		}
		if ts < 0 {
			ts = 0
		}
		out = append(out, ts)
	}
	return out
}

func formatTimestampSeconds(timestampMS int64) string {
	if timestampMS <= 0 {
		return "0"
	}
	return strconv.FormatFloat(float64(timestampMS)/1000, 'f', 3, 64)
}
