//go:build cgo

package llamacppnative

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeGemmaDescribesFixtureImages(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_GEMMA_DESCRIPTION_TEST") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_GEMMA_DESCRIPTION_TEST=1 with LLAMA_NATIVE_GEMMA_MODEL_PATH and LLAMA_NATIVE_GEMMA_MMPROJ_PATH")
	}

	runtime := loadGemmaNativeRuntimeForTest(t)
	t.Cleanup(func() { runtime.Close() })

	root := findRepoRoot(t)
	fixtures := []struct {
		name          string
		requireAny    []string
		requireAnyAlt []string
	}{
		{name: "cat_1.jpg", requireAny: []string{"cat"}},
		{name: "dog_1.jpg", requireAny: []string{"dog"}},
		{name: "woman_2.jpg", requireAny: []string{"woman", "person", "girl", "female"}},
		{
			name:          "woman_office.jpg",
			requireAny:    []string{"woman", "person", "girl", "female"},
			requireAnyAlt: []string{"office", "desk", "computer", "laptop", "monitor"},
		},
	}

	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.name, func(t *testing.T) {
			imagePath := filepath.Join(root, "fixtures", "images", fixture.name)
			description, raw, err := runtime.DescribeImage(context.Background(), imagePath)
			if err != nil {
				t.Fatalf("describe image: %v", err)
			}
			if strings.TrimSpace(description.ShortDescription) == "" {
				t.Fatalf("short_description is empty; raw=%s", raw)
			}
			if len(description.Labels) == 0 {
				t.Fatalf("labels are empty; raw=%s", raw)
			}

			combined := normalizeDescriptionSearchText(description)
			t.Logf("description=%q labels=%v", description.ShortDescription, description.Labels)

			if !containsAnyDescriptionKeyword(combined, fixture.requireAny) {
				t.Fatalf("description for %s missing expected keywords %v; raw=%s", fixture.name, fixture.requireAny, raw)
			}
			if len(fixture.requireAnyAlt) > 0 && !containsAnyDescriptionKeyword(combined, fixture.requireAnyAlt) {
				t.Fatalf("description for %s missing expected context keywords %v; raw=%s", fixture.name, fixture.requireAnyAlt, raw)
			}
		})
	}
}

func TestNativeGemmaAutotagsFixtureImages(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_GEMMA_AUTOTAG_TEST") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_GEMMA_AUTOTAG_TEST=1 with LLAMA_NATIVE_GEMMA_MODEL_PATH and LLAMA_NATIVE_GEMMA_MMPROJ_PATH")
	}

	runtime := loadGemmaNativeRuntimeForTest(t)
	t.Cleanup(func() { runtime.Close() })

	root := findRepoRoot(t)
	fixtures := []struct {
		name          string
		requireAny    []string
		requireAnyAlt []string
	}{
		{name: "cat_1.jpg", requireAny: []string{"cat"}},
		{name: "dog_1.jpg", requireAny: []string{"dog", "retriever"}},
		{name: "woman_2.jpg", requireAny: []string{"woman", "person", "female"}},
		{
			name:          "woman_office.jpg",
			requireAny:    []string{"woman", "person", "female"},
			requireAnyAlt: []string{"office", "desk", "professional", "business"},
		},
	}

	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.name, func(t *testing.T) {
			imagePath := filepath.Join(root, "fixtures", "images", fixture.name)
			tags, raw, err := runtime.AutoTagImage(context.Background(), imagePath)
			if err != nil {
				t.Fatalf("autotag image: %v", err)
			}
			if len(tags.Tags) < 3 || len(tags.Tags) > 10 {
				t.Fatalf("tags length out of range: %d raw=%s", len(tags.Tags), raw)
			}

			combined := strings.Join(tags.Tags, "\n")
			t.Logf("tags=%v", tags.Tags)

			if !containsAnyDescriptionKeyword(combined, fixture.requireAny) {
				t.Fatalf("tags for %s missing expected keywords %v; raw=%s", fixture.name, fixture.requireAny, raw)
			}
			if len(fixture.requireAnyAlt) > 0 && !containsAnyDescriptionKeyword(combined, fixture.requireAnyAlt) {
				t.Fatalf("tags for %s missing expected context keywords %v; raw=%s", fixture.name, fixture.requireAnyAlt, raw)
			}
		})
	}
}

func TestNativeGemmaDescribesAndTagsFixtureImages(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_GEMMA_DESCRIPTION_TAGS_TEST") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_GEMMA_DESCRIPTION_TAGS_TEST=1 with LLAMA_NATIVE_GEMMA_MODEL_PATH and LLAMA_NATIVE_GEMMA_MMPROJ_PATH")
	}

	runtime := loadGemmaNativeRuntimeForTest(t)
	t.Cleanup(func() { runtime.Close() })

	root := findRepoRoot(t)
	fixtures := []struct {
		name          string
		requireAny    []string
		requireAnyAlt []string
	}{
		{name: "cat_1.jpg", requireAny: []string{"cat"}},
		{name: "dog_1.jpg", requireAny: []string{"dog", "retriever"}},
		{name: "woman_2.jpg", requireAny: []string{"woman", "person", "female"}},
		{
			name:          "woman_office.jpg",
			requireAny:    []string{"woman", "person", "female"},
			requireAnyAlt: []string{"office", "desk", "professional", "business"},
		},
	}

	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.name, func(t *testing.T) {
			imagePath := filepath.Join(root, "fixtures", "images", fixture.name)
			result, raw, err := runtime.DescribeAndTagImage(context.Background(), imagePath)
			if err != nil {
				t.Fatalf("describe and tag image: %v", err)
			}
			if len(strings.Fields(result.Description)) < 16 {
				t.Fatalf("description is too short for %s: %q raw=%s", fixture.name, result.Description, raw)
			}
			if len(result.Tags) < 3 || len(result.Tags) > 10 {
				t.Fatalf("tags length out of range: %d raw=%s", len(result.Tags), raw)
			}

			combined := strings.ToLower(result.Description) + "\n" + strings.Join(result.Tags, "\n")
			t.Logf("description=%q tags=%v", result.Description, result.Tags)

			if !containsAnyDescriptionKeyword(combined, fixture.requireAny) {
				t.Fatalf("combined output for %s missing expected keywords %v; raw=%s", fixture.name, fixture.requireAny, raw)
			}
			if len(fixture.requireAnyAlt) > 0 && !containsAnyDescriptionKeyword(combined, fixture.requireAnyAlt) {
				t.Fatalf("combined output for %s missing expected context keywords %v; raw=%s", fixture.name, fixture.requireAnyAlt, raw)
			}
		})
	}
}

func loadGemmaNativeRuntimeForTest(t *testing.T) *nativeGemmaRuntime {
	t.Helper()

	modelPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_GEMMA_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set LLAMA_NATIVE_GEMMA_MODEL_PATH to the Gemma GGUF model path")
	}
	visionPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_GEMMA_MMPROJ_PATH"))
	if visionPath == "" {
		t.Skip("set LLAMA_NATIVE_GEMMA_MMPROJ_PATH to the Gemma mmproj GGUF path")
	}

	runtime, err := newGemmaNativeRuntime(nativeGemmaRuntimeConfig{
		ModelPath:       modelPath,
		VisionModelPath: visionPath,
		GPULayers:       envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_GPU_LAYERS", 99),
		UseGPU:          envBoolOrDefault("LLAMA_NATIVE_GEMMA_USE_GPU", true),
		ContextSize:     envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_CONTEXT_SIZE", 8192),
		BatchSize:       envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_BATCH_SIZE", 512),
		Threads:         envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_THREADS", 0),
		ImageMaxSide:    envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_IMAGE_MAX_SIDE", 1024),
		ImageMaxTokens:  envIntOrDefault(t, "LLAMA_NATIVE_GEMMA_IMAGE_MAX_TOKENS", 0),
	})
	if err != nil {
		t.Fatalf("new native Gemma runtime: %v", err)
	}

	return runtime
}

func normalizeDescriptionSearchText(description gemmaImageDescription) string {
	parts := []string{strings.ToLower(strings.TrimSpace(description.ShortDescription))}
	for _, label := range description.Labels {
		trimmed := strings.ToLower(strings.TrimSpace(label))
		if trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	return strings.Join(parts, "\n")
}

func containsAnyDescriptionKeyword(haystack string, keywords []string) bool {
	for _, keyword := range keywords {
		if strings.Contains(haystack, strings.ToLower(keyword)) {
			return true
		}
	}
	return false
}
