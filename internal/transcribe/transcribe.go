package transcribe

import (
	"context"
	"errors"
)

var ErrNoAudio = errors.New("video has no audio")
var ErrAudioTooLong = errors.New("audio exceeds maximum duration")

type Transcript struct {
	Text string
}

type VideoTranscriber interface {
	TranscribeVideo(ctx context.Context, videoPath string) (Transcript, error)
}
