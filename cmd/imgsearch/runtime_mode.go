package main

import (
	"fmt"
	"strings"
)

type runtimeMode string

const (
	runtimeModeAll    runtimeMode = "all"
	runtimeModeAPI    runtimeMode = "api"
	runtimeModeWorker runtimeMode = "worker"
)

func resolveRuntimeMode(raw string) (runtimeMode, error) {
	mode := runtimeMode(strings.ToLower(strings.TrimSpace(raw)))
	if mode == "" {
		mode = runtimeModeAll
	}
	switch mode {
	case runtimeModeAll, runtimeModeAPI, runtimeModeWorker:
		return mode, nil
	default:
		return "", fmt.Errorf("unsupported mode %q (expected %q, %q, or %q)", raw, runtimeModeAll, runtimeModeAPI, runtimeModeWorker)
	}
}

func (m runtimeMode) startsHTTP() bool {
	return m == runtimeModeAll || m == runtimeModeAPI
}

func (m runtimeMode) startsWorker() bool {
	return m == runtimeModeAll || m == runtimeModeWorker
}

func shouldLoadAnnotator(mode runtimeMode, enableAnnotations bool) bool {
	return enableAnnotations && mode.startsWorker()
}

func annotationModeWarning(mode runtimeMode, enableAnnotations bool, modelPath string, mmprojPath string) string {
	if mode == runtimeModeAPI && enableAnnotations {
		if strings.TrimSpace(modelPath) != "" || strings.TrimSpace(mmprojPath) != "" {
			return "annotations enabled but no worker is running in api mode; explicit annotator settings will be ignored"
		}
		return "annotations enabled but no worker is running in api mode; annotator settings will be ignored"
	}
	return ""
}
