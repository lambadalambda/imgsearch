package main

import "testing"

func TestResolveRuntimeModeDefaultsToAll(t *testing.T) {
	mode, err := resolveRuntimeMode("")
	if err != nil {
		t.Fatalf("resolve mode: %v", err)
	}
	if mode != runtimeModeAll {
		t.Fatalf("mode: got=%q want=%q", mode, runtimeModeAll)
	}
}

func TestResolveRuntimeModeParsesValidValues(t *testing.T) {
	apiMode, err := resolveRuntimeMode("api")
	if err != nil {
		t.Fatalf("resolve api mode: %v", err)
	}
	if !apiMode.startsHTTP() || apiMode.startsWorker() {
		t.Fatalf("unexpected api mode behavior: http=%t worker=%t", apiMode.startsHTTP(), apiMode.startsWorker())
	}

	workerMode, err := resolveRuntimeMode("worker")
	if err != nil {
		t.Fatalf("resolve worker mode: %v", err)
	}
	if workerMode.startsHTTP() || !workerMode.startsWorker() {
		t.Fatalf("unexpected worker mode behavior: http=%t worker=%t", workerMode.startsHTTP(), workerMode.startsWorker())
	}
}

func TestResolveRuntimeModeRejectsUnknownValue(t *testing.T) {
	if _, err := resolveRuntimeMode("mystery"); err == nil {
		t.Fatal("expected invalid mode error")
	}
}

func TestAnnotationModeHelpers(t *testing.T) {
	if shouldLoadAnnotator(runtimeModeAPI, true) {
		t.Fatal("expected api mode to skip annotator loading")
	}
	if warning := annotationModeWarning(runtimeModeAPI, true, "", ""); warning == "" {
		t.Fatal("expected api mode annotation warning")
	}
	if !shouldLoadAnnotator(runtimeModeWorker, true) {
		t.Fatal("expected worker mode to load annotator when enabled")
	}
	if warning := annotationModeWarning(runtimeModeWorker, true, "", ""); warning != "" {
		t.Fatalf("unexpected worker mode warning: %q", warning)
	}
}

func TestAnnotationModeWarningMentionsExplicitAnnotatorSettings(t *testing.T) {
	warning := annotationModeWarning(runtimeModeAPI, true, "/models/custom.gguf", "")
	if warning == "" || warning == "annotations enabled but no worker is running in api mode; annotator settings will be ignored" {
		t.Fatalf("expected specific explicit-settings warning, got %q", warning)
	}
}
