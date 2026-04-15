package main

import (
	"context"
	"fmt"
	"strings"
)

func resolveAnnotatorAssetPaths(
	ctx context.Context,
	enabled bool,
	shouldLoad bool,
	variant string,
	modelPath string,
	mmprojPath string,
	resolveDefaults func(context.Context, string, string, string) (string, string, error),
) (string, string, error) {
	trimmedModelPath := strings.TrimSpace(modelPath)
	trimmedMMProjPath := strings.TrimSpace(mmprojPath)
	if !enabled {
		if trimmedModelPath != "" || trimmedMMProjPath != "" {
			return "", "", fmt.Errorf("conflicting annotator flags: -enable-annotations=false but explicit annotator paths were provided")
		}
		return "", "", nil
	}
	if !shouldLoad {
		return "", "", nil
	}
	if trimmedModelPath != "" || trimmedMMProjPath != "" {
		return trimmedModelPath, trimmedMMProjPath, nil
	}
	return resolveDefaults(ctx, variant, trimmedModelPath, trimmedMMProjPath)
}
