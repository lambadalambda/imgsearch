package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
)

func llamaNativeModelVersion(modelPath string, visionModelPath string, dimensions int, imageMaxSide int, imageMaxTokens int, queryInstruction string, passageInstruction string) string {
	modelToken := sanitizeModelVersionToken(filepath.Base(strings.TrimSpace(modelPath)))
	visionToken := sanitizeModelVersionToken(filepath.Base(strings.TrimSpace(visionModelPath)))
	queryToken := instructionVersionToken(queryInstruction)
	passageToken := instructionVersionToken(passageInstruction)

	return fmt.Sprintf(
		"native-v5-chatml-vipsjpeg-%s-%s-d%d-s%d-t%d-q%s-p%s",
		modelToken,
		visionToken,
		dimensions,
		imageMaxSide,
		imageMaxTokens,
		queryToken,
		passageToken,
	)
}

func instructionVersionToken(value string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(value)))
	return hex.EncodeToString(sum[:6])
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
