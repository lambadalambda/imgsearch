package transcribe

import (
	"context"
	"errors"
)

var ErrNoAudio = errors.New("video has no audio")

type Transcript struct {
	Text string
}

type VideoTranscriber interface {
	TranscribeVideo(ctx context.Context, videoPath string) (Transcript, error)
}
