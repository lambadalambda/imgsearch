package jinamlx

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestSidecarEmbeddingsAreSemanticallyReasonable(t *testing.T) {
	if os.Getenv("RUN_JINA_MLX_INTEGRATION") != "1" {
		t.Skip("set RUN_JINA_MLX_INTEGRATION=1 and run `mise run jina-serve` or `mise run jina-torch-serve` to enable")
	}

	baseURL := os.Getenv("JINA_MLX_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:9009"
	}

	if err := waitForSidecar(baseURL, 90*time.Second); err != nil {
		t.Fatalf("sidecar not ready at %s: %v", baseURL, err)
	}

	repoRoot := findRepoRoot(t)
	staged := stageFixturesForSidecar(t, repoRoot, []string{
		"cat_1.jpg",
		"cat_2.webp",
		"dog_2.avif",
		"woman_2.jpg",
		"woman_office.jpg",
	})

	client := NewHTTPClientWithImageMode(baseURL, string(ImageModeAuto))
	vectors := map[string][]float32{}
	for name, path := range staged {
		vec, err := client.EmbedImage(context.Background(), path)
		if err != nil {
			t.Fatalf("embed %s: %v", name, err)
		}
		if len(vec) == 0 {
			t.Fatalf("embed %s returned empty vector", name)
		}
		vectors[name] = vec
	}

	catCat := cosine(vectors["cat_1.jpg"], vectors["cat_2.webp"])
	catDog := cosine(vectors["cat_1.jpg"], vectors["dog_2.avif"])
	catWoman := cosine(vectors["cat_1.jpg"], vectors["woman_2.jpg"])
	womanWoman := cosine(vectors["woman_2.jpg"], vectors["woman_office.jpg"])
	womanDog := cosine(vectors["woman_2.jpg"], vectors["dog_2.avif"])

	if !(catCat > catDog && catCat > catWoman) {
		t.Fatalf("expected cat pair to be closer than cat cross-category pairs: cat-cat=%.4f cat-dog=%.4f cat-woman=%.4f", catCat, catDog, catWoman)
	}
	if womanWoman <= womanDog {
		t.Fatalf("expected women pair to be closer than woman-dog: women=%.4f woman-dog=%.4f", womanWoman, womanDog)
	}

	within := (catCat + womanWoman) / 2
	cross := (catDog + catWoman + womanDog) / 3
	if within <= cross {
		t.Fatalf("expected within-category similarity above cross-category: within=%.4f cross=%.4f", within, cross)
	}
}

func TestSidecarTextEmbeddingIsStableAcrossAlternatingQueries(t *testing.T) {
	if os.Getenv("RUN_JINA_MLX_INTEGRATION") != "1" {
		t.Skip("set RUN_JINA_MLX_INTEGRATION=1 and run `mise run jina-serve` or `mise run jina-torch-serve` to enable")
	}

	baseURL := os.Getenv("JINA_MLX_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:9009"
	}

	if err := waitForSidecar(baseURL, 90*time.Second); err != nil {
		t.Fatalf("sidecar not ready at %s: %v", baseURL, err)
	}

	client := NewHTTPClientWithImageMode(baseURL, string(ImageModeAuto))
	baselineWoman, err := client.EmbedText(context.Background(), "woman")
	if err != nil {
		t.Fatalf("baseline woman embedding: %v", err)
	}
	baselineCat, err := client.EmbedText(context.Background(), "cat")
	if err != nil {
		t.Fatalf("baseline cat embedding: %v", err)
	}

	for i := 0; i < 120; i++ {
		query := "woman"
		baseline := baselineWoman
		if i%2 == 1 {
			query = "cat"
			baseline = baselineCat
		}

		vec, err := client.EmbedText(context.Background(), query)
		if err != nil {
			t.Fatalf("embed query %q at iter=%d: %v", query, i, err)
		}
		if sim := cosine(vec, baseline); sim < 0.999999 {
			t.Fatalf("unstable embedding for %q at iter=%d: cosine=%.8f", query, i, sim)
		}
	}
}

func waitForSidecar(baseURL string, timeout time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(baseURL + "/healthz")
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for healthy sidecar")
}

func stageFixturesForSidecar(t *testing.T, repoRoot string, names []string) map[string]string {
	t.Helper()

	allowedDir := filepath.Join(repoRoot, "data", "images")
	if err := os.MkdirAll(allowedDir, 0o755); err != nil {
		t.Fatalf("mkdir allowed dir: %v", err)
	}

	out := make(map[string]string, len(names))
	for i, name := range names {
		src := filepath.Join(repoRoot, "fixtures", "images", name)
		raw, err := os.ReadFile(src)
		if err != nil {
			t.Fatalf("read fixture %s: %v", name, err)
		}

		dstName := fmt.Sprintf("sidecar-integration-%d-%s", i, filepath.Base(name))
		dst := filepath.Join(allowedDir, dstName)
		if err := os.WriteFile(dst, raw, 0o644); err != nil {
			t.Fatalf("write staged fixture %s: %v", name, err)
		}
		t.Cleanup(func() { _ = os.Remove(dst) })
		out[name] = dst
	}

	return out
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
