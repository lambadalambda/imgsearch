//go:build llamacpp_native && cgo

package llamacppnative

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type fixtureExpectation struct {
	Query         string
	ExpectedNames []string
}

type scoredFixture struct {
	Name  string
	Score float64
}

var fixtureExpectationPattern = regexp.MustCompile(`^"(.+?)"\s*[:\-]\s*(.+)$`)

func TestNativeEmbeddingsAreSemanticallyReasonable(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_INTEGRATION") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_INTEGRATION=1 with LLAMA_NATIVE_MODEL_PATH and LLAMA_NATIVE_MMPROJ_PATH")
	}

	embedder := newNativeEmbedderForIntegration(t)

	repoRoot := findRepoRoot(t)
	fixtures := map[string]string{
		"cat":          filepath.Join(repoRoot, "fixtures", "images", "cat_1.jpg"),
		"dog":          filepath.Join(repoRoot, "fixtures", "images", "dog_1.jpg"),
		"woman":        filepath.Join(repoRoot, "fixtures", "images", "woman_2.jpg"),
		"woman_office": filepath.Join(repoRoot, "fixtures", "images", "woman_office.jpg"),
	}

	vectors := map[string][]float32{}
	for key, path := range fixtures {
		vec, err := embedder.EmbedImage(context.Background(), path)
		if err != nil {
			t.Fatalf("embed %s image: %v", key, err)
		}
		if len(vec) != embedder.dimensions {
			t.Fatalf("embedding dimensions for %s: got=%d want=%d", key, len(vec), embedder.dimensions)
		}
		vectors[key] = vec
	}

	womanWoman := cosine(vectors["woman"], vectors["woman_office"])
	womanDog := cosine(vectors["woman"], vectors["dog"])
	if womanWoman <= womanDog {
		t.Fatalf("expected woman pair similarity > woman-dog: woman-woman=%.4f woman-dog=%.4f", womanWoman, womanDog)
	}

	dogQuery, err := embedder.EmbedText(context.Background(), "a dog playing outdoors")
	if err != nil {
		t.Fatalf("embed dog query: %v", err)
	}
	if len(dogQuery) != embedder.dimensions {
		t.Fatalf("dog query embedding dimensions: got=%d want=%d", len(dogQuery), embedder.dimensions)
	}

	dogScore := cosine(dogQuery, vectors["dog"])
	catScore := cosine(dogQuery, vectors["cat"])
	womanScore := cosine(dogQuery, vectors["woman"])
	if dogScore <= catScore {
		t.Fatalf("expected dog query to rank dog above cat: dog=%.4f cat=%.4f woman=%.4f", dogScore, catScore, womanScore)
	}
}

func TestNativeFixtureRetrievalQuality(t *testing.T) {
	if os.Getenv("RUN_LLAMACPP_NATIVE_INTEGRATION") != "1" {
		t.Skip("set RUN_LLAMACPP_NATIVE_INTEGRATION=1 with LLAMA_NATIVE_MODEL_PATH and LLAMA_NATIVE_MMPROJ_PATH")
	}

	embedder := newNativeEmbedderForIntegration(t)

	root := findRepoRoot(t)
	fixturesDir := filepath.Join(root, "fixtures", "images")
	expectedPath := filepath.Join(fixturesDir, "expected.txt")

	expectedRaw, err := os.ReadFile(expectedPath)
	if err != nil {
		t.Fatalf("read expected fixture mapping: %v", err)
	}
	expectations := parseFixtureExpectations(string(expectedRaw))
	if len(expectations) == 0 {
		t.Fatalf("no fixture expectations parsed from %s", expectedPath)
	}

	entries, err := os.ReadDir(fixturesDir)
	if err != nil {
		t.Fatalf("read fixtures dir: %v", err)
	}
	imageNames := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if isFixtureImageName(name) {
			imageNames = append(imageNames, name)
		}
	}
	sort.Strings(imageNames)
	if len(imageNames) == 0 {
		t.Fatal("no fixture images found")
	}

	missingNames := missingExpectedFixtureNames(imageNames, expectations)
	if len(missingNames) > 0 {
		t.Fatalf("expected fixture names missing from fixtures/images: %s", strings.Join(missingNames, ", "))
	}

	vectors := make(map[string][]float32, len(imageNames))
	for _, name := range imageNames {
		path := filepath.Join(fixturesDir, name)
		vec, err := embedder.EmbedImage(context.Background(), path)
		if err != nil {
			t.Fatalf("embed fixture %s: %v", name, err)
		}
		if len(vec) != embedder.dimensions {
			t.Fatalf("embedding dimensions for %s: got=%d want=%d", name, len(vec), embedder.dimensions)
		}
		vectors[name] = vec
	}

	failures := make([]string, 0)
	for _, expectation := range expectations {
		queryVec, err := embedder.EmbedText(context.Background(), expectation.Query)
		if err != nil {
			t.Fatalf("embed query %q: %v", expectation.Query, err)
		}
		if len(queryVec) != embedder.dimensions {
			t.Fatalf("query embedding dimensions for %q: got=%d want=%d", expectation.Query, len(queryVec), embedder.dimensions)
		}

		scored := make([]scoredFixture, 0, len(imageNames))
		for _, name := range imageNames {
			scored = append(scored, scoredFixture{Name: name, Score: cosine(queryVec, vectors[name])})
		}
		sort.Slice(scored, func(i int, j int) bool {
			if scored[i].Score == scored[j].Score {
				return scored[i].Name < scored[j].Name
			}
			return scored[i].Score > scored[j].Score
		})

		hitCount := fixtureExpectationHitCount(scored, expectation.ExpectedNames)
		expectedCount := len(expectation.ExpectedNames)
		strictPass := hitCount == expectedCount
		relaxedPass := fixtureExpectationPasses(hitCount, expectedCount, false)

		topN := 5
		if len(scored) < topN {
			topN = len(scored)
		}
		expectedSet := make(map[string]struct{}, len(expectation.ExpectedNames))
		for _, expectedName := range expectation.ExpectedNames {
			expectedSet[expectedName] = struct{}{}
		}
		lines := make([]string, 0, topN)
		for i := 0; i < topN; i++ {
			marker := " "
			if _, ok := expectedSet[scored[i].Name]; ok {
				marker = "*"
			}
			lines = append(lines, fmt.Sprintf("%d. %s %.4f%s", i+1, scored[i].Name, scored[i].Score, marker))
		}
		t.Logf("query=%q expected=%v hits=%d/%d strict=%t relaxed=%t top%d=%s", expectation.Query, expectation.ExpectedNames, hitCount, expectedCount, strictPass, relaxedPass, topN, strings.Join(lines, " | "))

		if !relaxedPass {
			failures = append(failures, fmt.Sprintf("query %q expected hits %d/%d", expectation.Query, hitCount, expectedCount))
		}
	}

	if len(failures) > 0 {
		t.Fatalf("llama-cpp-native fixture retrieval failed: %s", strings.Join(failures, "; "))
	}
}

func newNativeEmbedderForIntegration(t *testing.T) *Embedder {
	t.Helper()

	modelPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set LLAMA_NATIVE_MODEL_PATH to llama.cpp GGUF embedding model")
	}
	visionPath := strings.TrimSpace(os.Getenv("LLAMA_NATIVE_MMPROJ_PATH"))
	if visionPath == "" {
		t.Skip("set LLAMA_NATIVE_MMPROJ_PATH to llama.cpp GGUF vision projector")
	}

	e, err := New(Config{
		ModelPath:          modelPath,
		VisionModelPath:    visionPath,
		Dimensions:         envIntOrDefault(t, "LLAMA_NATIVE_DIMS", 2048),
		GPULayers:          envIntOrDefault(t, "LLAMA_NATIVE_GPU_LAYERS", 99),
		UseGPU:             envBoolOrDefault("LLAMA_NATIVE_USE_GPU", true),
		ContextSize:        envIntOrDefault(t, "LLAMA_NATIVE_CONTEXT_SIZE", 8192),
		BatchSize:          envIntOrDefault(t, "LLAMA_NATIVE_BATCH_SIZE", 512),
		Threads:            envIntOrDefault(t, "LLAMA_NATIVE_THREADS", 0),
		ImageMaxSide:       envIntOrDefault(t, "LLAMA_NATIVE_IMAGE_MAX_SIDE", 512),
		ImageMaxTokens:     envIntOrDefault(t, "LLAMA_NATIVE_IMAGE_MAX_TOKENS", 0),
		QueryInstruction:   envOr("LLAMA_NATIVE_QUERY_INSTRUCTION", defaultQueryInstruction),
		PassageInstruction: envOr("LLAMA_NATIVE_PASSAGE_INSTRUCTION", defaultPassageInstruction),
	})
	if err != nil {
		t.Fatalf("new llama-cpp-native embedder: %v", err)
	}
	t.Cleanup(func() { _ = e.Close() })
	return e
}

func parseFixtureExpectations(contents string) []fixtureExpectation {
	out := make([]fixtureExpectation, 0)
	for _, rawLine := range strings.Split(contents, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "clusters by") {
			continue
		}

		match := fixtureExpectationPattern.FindStringSubmatch(line)
		if len(match) != 3 {
			continue
		}

		query := strings.TrimSpace(match[1])
		namesRaw := strings.Split(match[2], ",")
		names := make([]string, 0, len(namesRaw))
		for _, name := range namesRaw {
			trimmed := strings.TrimSpace(name)
			if trimmed != "" {
				names = append(names, trimmed)
			}
		}

		if query != "" && len(names) > 0 {
			out = append(out, fixtureExpectation{Query: query, ExpectedNames: names})
		}
	}
	return out
}

func fixtureExpectationHitCount(scored []scoredFixture, expectedNames []string) int {
	k := len(expectedNames)
	if k <= 0 {
		return 0
	}
	if k > len(scored) {
		k = len(scored)
	}

	expectedSet := make(map[string]struct{}, len(expectedNames))
	for _, name := range expectedNames {
		expectedSet[name] = struct{}{}
	}

	hits := 0
	for _, row := range scored[:k] {
		if _, ok := expectedSet[row.Name]; ok {
			hits++
		}
	}
	return hits
}

func fixtureExpectationPasses(hitCount int, expectedCount int, strict bool) bool {
	if expectedCount <= 0 {
		return false
	}
	if strict {
		return hitCount == expectedCount
	}
	threshold := expectedCount - 1
	if threshold < 1 {
		threshold = 1
	}
	return hitCount >= threshold
}

func isFixtureImageName(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".jpg") ||
		strings.HasSuffix(lower, ".jpeg") ||
		strings.HasSuffix(lower, ".png")
}

func missingExpectedFixtureNames(imageNames []string, expectations []fixtureExpectation) []string {
	present := make(map[string]struct{}, len(imageNames))
	for _, name := range imageNames {
		present[name] = struct{}{}
	}

	missingSet := map[string]struct{}{}
	for _, expectation := range expectations {
		for _, name := range expectation.ExpectedNames {
			if _, ok := present[name]; !ok {
				missingSet[name] = struct{}{}
			}
		}
	}

	missing := make([]string, 0, len(missingSet))
	for name := range missingSet {
		missing = append(missing, name)
	}
	sort.Strings(missing)
	return missing
}

func envOr(key string, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func envIntOrDefault(t *testing.T, key string, fallback int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("parse %s as int: %v", key, err)
	}
	return n
}

func envBoolOrDefault(key string, fallback bool) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if v == "" {
		return fallback
	}
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func findRepoRoot(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine caller path")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repository root")
		}
		dir = parent
	}
}

func cosine(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	var dot float64
	var na float64
	var nb float64
	for i := 0; i < n; i++ {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
