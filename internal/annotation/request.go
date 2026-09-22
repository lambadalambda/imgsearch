package annotation

import "imgsearch/internal/embedder"

// Request bundles everything a backend needs to ask for one annotation: the
// primary prompt pair and the compact retry pair used when the first answer
// does not parse.
type Request struct {
	SystemPrompt      string
	UserPrompt        string
	JSONSchema        string
	MaxTokens         int
	RetrySystemPrompt string
	RetryUserPrompt   string
	RetryMaxTokens    int
}

// ImageRequest is the rich standalone-image annotation.
func ImageRequest(originalName string, knownTags []string) Request {
	return Request{
		SystemPrompt:      ImageSystemPrompt,
		UserPrompt:        ImageUserPrompt(originalName, knownTags),
		JSONSchema:        FullJSONSchema,
		MaxTokens:         ImageMaxTokens,
		RetrySystemPrompt: ImageRetrySystemPrompt,
		RetryUserPrompt:   ImageRetryUserPrompt,
		RetryMaxTokens:    ImageRetryMaxTokens,
	}
}

// VideoFrameRequest is the compact per-frame annotation.
func VideoFrameRequest(originalName string, knownTags []string) Request {
	return Request{
		SystemPrompt:      VideoFrameSystemPrompt,
		UserPrompt:        VideoFrameUserPrompt(originalName, knownTags),
		JSONSchema:        CompactJSONSchema,
		MaxTokens:         VideoFrameMaxTokens,
		RetrySystemPrompt: VideoFrameRetrySystemPrompt,
		RetryUserPrompt:   VideoFrameRetryUserPrompt,
		RetryMaxTokens:    VideoFrameRetryMaxTokens,
	}
}

// VideoRequest is the video-level summary built from frame evidence. The
// backend still attaches the representative frame image.
func VideoRequest(input embedder.VideoAnnotationInput) (Request, error) {
	userPrompt, err := VideoUserPrompt(input)
	if err != nil {
		return Request{}, err
	}
	return Request{
		SystemPrompt:      VideoSystemPrompt,
		UserPrompt:        userPrompt,
		JSONSchema:        FullJSONSchema,
		MaxTokens:         VideoMaxTokens,
		RetrySystemPrompt: VideoRetrySystemPrompt,
		RetryUserPrompt:   VideoRetryUserPrompt,
		RetryMaxTokens:    VideoRetryMaxTokens,
	}, nil
}
