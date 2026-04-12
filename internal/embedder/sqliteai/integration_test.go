package sqliteai

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	sqlite3 "github.com/mattn/go-sqlite3"
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

func TestParseFixtureExpectationsSupportsDashAndColon(t *testing.T) {
	text := strings.Join([]string{
		"Clusters by text query:",
		"",
		"\"cat\" - cat_1.jpg, cat_2.jpg",
		"\"woman in office\": woman_office.jpg",
		"\"how are you feeling today?\": baileys.png, guiness.png",
	}, "\n")

	got := parseFixtureExpectations(text)
	if len(got) != 3 {
		t.Fatalf("expectation count: got=%d want=3", len(got))
	}

	if got[0].Query != "cat" {
		t.Fatalf("first query: got=%q want=cat", got[0].Query)
	}
	if strings.Join(got[0].ExpectedNames, ",") != "cat_1.jpg,cat_2.jpg" {
		t.Fatalf("first expected names: got=%v", got[0].ExpectedNames)
	}

	if got[1].Query != "woman in office" {
		t.Fatalf("second query: got=%q want=woman in office", got[1].Query)
	}
	if len(got[1].ExpectedNames) != 1 || got[1].ExpectedNames[0] != "woman_office.jpg" {
		t.Fatalf("second expected names: got=%v", got[1].ExpectedNames)
	}
}

func TestFixtureExpectationHitCountUsesTopK(t *testing.T) {
	scored := []scoredFixture{
		{Name: "a.jpg", Score: 0.9},
		{Name: "b.jpg", Score: 0.8},
		{Name: "c.jpg", Score: 0.7},
		{Name: "d.jpg", Score: 0.6},
	}
	hitCount := fixtureExpectationHitCount(scored, []string{"a.jpg", "d.jpg"})
	if hitCount != 1 {
		t.Fatalf("hit count: got=%d want=1", hitCount)
	}
}

func TestFixtureExpectationPassesRelaxedAndStrict(t *testing.T) {
	if !fixtureExpectationPasses(2, 2, true) {
		t.Fatal("expected strict pass for perfect hits")
	}
	if fixtureExpectationPasses(1, 2, true) {
		t.Fatal("expected strict failure when not all expected names hit")
	}
	if !fixtureExpectationPasses(3, 4, false) {
		t.Fatal("expected relaxed pass for one miss in four")
	}
	if fixtureExpectationPasses(2, 4, false) {
		t.Fatal("expected relaxed failure for two misses in four")
	}
}

func TestSQLiteAIEmbeddingsAreSemanticallyReasonable(t *testing.T) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		t.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := os.Getenv("SQLITE_AI_PATH")
	if extensionPath == "" {
		t.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := os.Getenv("SQLITE_AI_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := os.Getenv("SQLITE_AI_VISION_PATH")
	if visionPath == "" {
		t.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dbConn := openWithSQLiteAI(t, extensionPath)

	modelOptions := envOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99")
	visionOptions := envOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1")
	contextOptions := envOr("SQLITE_AI_CONTEXT_OPTIONS", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=last")
	expectedDimensions := envIntOrDefault("SQLITE_AI_DIMS", 4096)

	embedder, err := New(Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       modelOptions,
		VisionModelPath:    visionPath,
		VisionModelOptions: visionOptions,
		ContextOptions:     contextOptions,
	})
	if err != nil {
		t.Fatalf("new sqlite-ai embedder: %v", err)
	}

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
		if len(vec) != expectedDimensions {
			t.Fatalf("embedding dimensions for %s: got=%d want=%d", key, len(vec), expectedDimensions)
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
	if len(dogQuery) != expectedDimensions {
		t.Fatalf("dog query embedding dimensions: got=%d want=%d", len(dogQuery), expectedDimensions)
	}

	dogScore := cosine(dogQuery, vectors["dog"])
	catScore := cosine(dogQuery, vectors["cat"])
	womanScore := cosine(dogQuery, vectors["woman"])
	if dogScore <= catScore {
		t.Fatalf("expected dog query to rank dog above cat: dog=%.4f cat=%.4f woman=%.4f", dogScore, catScore, womanScore)
	}
}

func TestSQLiteAIFixtureRetrievalQuality(t *testing.T) {
	if os.Getenv("RUN_SQLITE_AI_INTEGRATION") != "1" {
		t.Skip("set RUN_SQLITE_AI_INTEGRATION=1 with SQLITE_AI_PATH, SQLITE_AI_MODEL_PATH, and SQLITE_AI_VISION_PATH")
	}
	if _, err := exec.LookPath("vips"); err != nil {
		t.Skip("vips CLI is required for sqlite-ai image preprocessing")
	}

	extensionPath := os.Getenv("SQLITE_AI_PATH")
	if extensionPath == "" {
		t.Skip("set SQLITE_AI_PATH to sqlite-ai extension binary path")
	}
	modelPath := os.Getenv("SQLITE_AI_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set SQLITE_AI_MODEL_PATH to sqlite-ai GGUF embedding model")
	}
	visionPath := os.Getenv("SQLITE_AI_VISION_PATH")
	if visionPath == "" {
		t.Skip("set SQLITE_AI_VISION_PATH to sqlite-ai GGUF vision projector")
	}

	dbConn := openWithSQLiteAI(t, extensionPath)

	embedder, err := New(Config{
		DB:                 dbConn,
		ModelPath:          modelPath,
		ModelOptions:       envOr("SQLITE_AI_MODEL_OPTIONS", "gpu_layers=99"),
		VisionModelPath:    visionPath,
		VisionModelOptions: envOr("SQLITE_AI_VISION_OPTIONS", "use_gpu=1"),
		ContextOptions:     envOr("SQLITE_AI_CONTEXT_OPTIONS", "embedding_type=FLOAT32,normalize_embedding=1,pooling_type=last"),
		QueryInstruction:   envOr("SQLITE_AI_QUERY_INSTRUCTION", "Retrieve images or text relevant to the user's query."),
		PassageInstruction: envOr("SQLITE_AI_PASSAGE_INSTRUCTION", "Represent this image or text for retrieval."),
		ImageMaxSide:       envIntOrDefault("SQLITE_AI_MAX_SIDE", 512),
		VipsPath:           envOr("SQLITE_AI_VIPS_PATH", "vips"),
	})
	if err != nil {
		t.Fatalf("new sqlite-ai embedder: %v", err)
	}

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

	expectedDimensions := envIntOrDefault("SQLITE_AI_DIMS", 4096)
	vectors := make(map[string][]float32, len(imageNames))
	for _, name := range imageNames {
		path := filepath.Join(fixturesDir, name)
		vec, err := embedder.EmbedImage(context.Background(), path)
		if err != nil {
			t.Fatalf("embed fixture %s: %v", name, err)
		}
		if len(vec) != expectedDimensions {
			t.Fatalf("embedding dimensions for %s: got=%d want=%d", name, len(vec), expectedDimensions)
		}
		vectors[name] = vec
	}

	failures := make([]string, 0)
	for _, expectation := range expectations {
		queryVec, err := embedder.EmbedText(context.Background(), expectation.Query)
		if err != nil {
			t.Fatalf("embed query %q: %v", expectation.Query, err)
		}
		if len(queryVec) != expectedDimensions {
			t.Fatalf("query embedding dimensions for %q: got=%d want=%d", expectation.Query, len(queryVec), expectedDimensions)
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
		t.Fatalf("sqlite-ai fixture retrieval failed: %s", strings.Join(failures, "; "))
	}
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
	return hitCount >= max(1, expectedCount-1)
}

func isFixtureImageName(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".jpg") ||
		strings.HasSuffix(lower, ".jpeg") ||
		strings.HasSuffix(lower, ".png") ||
		strings.HasSuffix(lower, ".webp") ||
		strings.HasSuffix(lower, ".avif")
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

func envIntOrDefault(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

var sqliteAIDriverCounter uint64

func openWithSQLiteAI(t *testing.T, extensionPath string) *sql.DB {
	t.Helper()

	abs, err := filepath.Abs(extensionPath)
	if err != nil {
		t.Fatalf("resolve extension path: %v", err)
	}
	loadTarget := abs
	ext := strings.ToLower(filepath.Ext(loadTarget))
	if ext == ".dylib" || ext == ".so" || ext == ".dll" {
		loadTarget = strings.TrimSuffix(loadTarget, filepath.Ext(loadTarget))
	}

	driverName := fmt.Sprintf("sqlite3_with_ai_test_%d", atomic.AddUint64(&sqliteAIDriverCounter, 1))
	sql.Register(driverName, &sqlite3.SQLiteDriver{
		ConnectHook: func(conn *sqlite3.SQLiteConn) error {
			if err := conn.LoadExtension(loadTarget, "sqlite3_ai_init"); err != nil {
				return fmt.Errorf("load sqlite-ai extension from %q: %w", loadTarget, err)
			}
			return nil
		},
	})

	dbConn, err := sql.Open(driverName, ":memory:")
	if err != nil {
		t.Fatalf("open sqlite db: %v", err)
	}
	t.Cleanup(func() { _ = dbConn.Close() })
	return dbConn
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
