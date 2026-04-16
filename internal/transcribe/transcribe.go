package transcribe

import "context"

type Transcript struct {
	Text string
}

type VideoTranscriber interface {
	TranscribeVideo(ctx context.Context, videoPath string) (Transcript, error)
}
