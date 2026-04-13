package main

import (
	"fmt"
	"path/filepath"
	"strings"
)

func llamaCPPModelVersion(model string, dimensions int) string {
	modelToken := sanitizeModelVersionToken(strings.TrimSpace(model))
	if modelToken == "unknown" {
		modelToken = "default"
	}

	return fmt.Sprintf("server-v2-%s-d%d", modelToken, dimensions)
}

func llamaNativeModelVersion(modelPath string, visionModelPath string, dimensions int, imageMaxSide int, imageMaxTokens int) string {
	modelToken := sanitizeModelVersionToken(filepath.Base(strings.TrimSpace(modelPath)))
	visionToken := sanitizeModelVersionToken(filepath.Base(strings.TrimSpace(visionModelPath)))

	return fmt.Sprintf(
		"native-v3-vipsjpeg-%s-%s-d%d-s%d-t%d",
		modelToken,
		visionToken,
		dimensions,
		imageMaxSide,
		imageMaxTokens,
	)
}

func sanitizeModelVersionToken(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return "unknown"
	}

	b := strings.Builder{}
	b.Grow(len(trimmed))
	for _, r := range trimmed {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '.', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}

	sanitized := strings.Trim(b.String(), "._-")
	if sanitized == "" {
		return "unknown"
	}

	return sanitized
}
