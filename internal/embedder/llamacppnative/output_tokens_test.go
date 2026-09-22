//go:build cgo

package llamacppnative

import (
	"testing"

	"imgsearch/internal/annotation"
	"imgsearch/internal/embedder"
)

func TestNativeAnnotationOutputTokensPrimaryAndRetry(t *testing.T) {
	for _, custom := range []bool{false, true} {
		pair := embedder.AnnotationTokenBudget{}
		if custom {
			pair = embedder.AnnotationTokenBudget{Primary: 2100, Retry: 800}
		}
		video, err := annotation.VideoRequest(embedder.VideoAnnotationInput{Frames: []embedder.VideoFrameAnnotation{{Description: "frame"}}, OutputTokens: pair})
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name           string
			req            annotation.Request
			primary, retry int
		}{
			{"image", annotation.ImageRequest("", nil).WithOutputTokens(pair), 1024, 512},
			{"video_frame", annotation.VideoFrameRequest("", nil).WithOutputTokens(pair), 320, 256},
			{"video", video, 1200, 640},
		} {
			name := tc.name + "/defaults"
			if custom {
				name = tc.name + "/custom"
				tc.primary = pair.Primary
				tc.retry = pair.Retry
			}
			t.Run(name, func(t *testing.T) {
				calls := 0
				result, _, err := describeAndTagImageWithGenerator(tc.req, func(attempt, system, user, schema string, maxTokens int) (string, error) {
					calls++
					if schema != tc.req.JSONSchema {
						t.Error("schema changed")
					}
					if calls == 1 {
						if attempt != "primary" || system != tc.req.SystemPrompt || user != tc.req.UserPrompt || maxTokens != tc.primary {
							t.Errorf("primary: %s tokens=%d want=%d", attempt, maxTokens, tc.primary)
						}
						return "not JSON", nil
					}
					if attempt != "retry" || system != tc.req.RetrySystemPrompt || user != tc.req.RetryUserPrompt || maxTokens != tc.retry {
						t.Errorf("retry: %s tokens=%d want=%d", attempt, maxTokens, tc.retry)
					}
					return `{"title":"Cat","description":"A cat","tags":["cat"]}`, nil
				})
				if err != nil || calls != 2 || result.Description != "A cat" {
					t.Fatalf("result=%+v calls=%d err=%v", result, calls, err)
				}
			})
		}
	}
}
