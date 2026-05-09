package webui

import (
	"bytes"
	"embed"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

//go:embed static/*
var legacyFiles embed.FS

//go:embed all:atelier/dist
var atelierFiles embed.FS

const atelierMissingMessage = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>imgsearch</title>
<style>
  body { font: 16px/1.55 -apple-system, "Inter", system-ui, sans-serif; background: #faf7f2; color: #1f1c17; margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 24px; }
  main { max-width: 540px; }
  h1 { font: 600 28px/1.15 "Iowan Old Style", "Source Serif Pro", Georgia, serif; margin: 0 0 12px; letter-spacing: -0.012em; }
  p { margin: 0 0 12px; color: #6c645a; }
  code { background: #fff; border: 1px solid #e7e0d2; border-radius: 6px; padding: 1px 6px; font: 13px/1 ui-monospace, "SF Mono", Menlo, monospace; }
  a { color: #c2492a; }
</style>
</head>
<body>
  <main>
    <h1>Atelier frontend not built yet</h1>
    <p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code> (or <code>mise run build:frontend</code>) to populate <code>internal/webui/atelier/dist</code>.</p>
    <p>The legacy UI is still available at <a href="/legacy">/legacy</a>.</p>
  </main>
</body>
</html>
`

// NewHandler returns the HTTP handler for the imgsearch web shell. The new
// Atelier SPA is served at "/" (with a tiny "not built" fallback when the
// embedded dist is empty), and the legacy single-file UI is preserved at
// "/legacy" so unported features remain reachable while we port them.
func NewHandler(dataDir string) http.Handler {
	if dataDir == "" {
		dataDir = "."
	}

	legacyAssets, err := fs.Sub(legacyFiles, "static")
	if err != nil {
		panic("webui legacy assets unavailable: " + err.Error())
	}
	legacyIndex, err := fs.ReadFile(legacyAssets, "index.html")
	if err != nil {
		panic("webui legacy index unavailable: " + err.Error())
	}

	atelierAssets, err := fs.Sub(atelierFiles, "atelier/dist")
	if err != nil {
		panic("webui atelier assets unavailable: " + err.Error())
	}
	atelierIndex, hasAtelier := readAtelierIndex(atelierAssets)

	imagesRoot := filepath.Join(dataDir, "images")
	videosRoot := filepath.Join(dataDir, "videos")
	mediaMux := http.NewServeMux()
	mediaMux.Handle("/images/", http.StripPrefix("/images/", http.FileServer(http.Dir(imagesRoot))))
	mediaMux.Handle("/videos/", http.StripPrefix("/videos/", videoFileServer(videosRoot)))

	mux := http.NewServeMux()
	mux.Handle("/media/", http.StripPrefix("/media", mediaMux))

	// Legacy UI at /legacy (assets at /legacy/assets/*).
	mux.Handle("/legacy/assets/", http.StripPrefix("/legacy/assets/", http.FileServer(http.FS(legacyAssets))))
	mux.HandleFunc("/legacy", legacyIndexHandler(legacyIndex))
	mux.HandleFunc("/legacy/", legacyIndexHandler(legacyIndex))

	// Atelier SPA at /. Vite emits files into dist/assets/, so we serve the
	// embedded FS at its root and let the URL path resolve to assets/* inside.
	mux.Handle("/assets/", http.FileServer(http.FS(atelierAssets)))
	mux.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFileFS(w, r, atelierAssets, "favicon.ico")
	})
	mux.HandleFunc("/", atelierIndexHandler(atelierIndex, hasAtelier))

	return mux
}

func readAtelierIndex(atelierAssets fs.FS) ([]byte, bool) {
	data, err := fs.ReadFile(atelierAssets, "index.html")
	if err != nil {
		return []byte(atelierMissingMessage), false
	}
	return data, true
}

func atelierIndexHandler(index []byte, _ bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// Only serve index for "/" — anything else under the SPA root falls through to 404,
		// which is fine because we use query-string routing rather than path-based history.
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(index)
	}
}

func legacyIndexHandler(index []byte) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(index)
	}
}

func videoFileServer(root string) http.Handler {
	files := http.FileServer(http.Dir(root))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			if ct := sniffStoredVideoContentType(root, r.URL.Path); ct != "" {
				w.Header().Set("Content-Type", ct)
			}
		}

		files.ServeHTTP(w, r)
	})
}

func sniffStoredVideoContentType(root string, requestPath string) string {
	cleanPath := path.Clean("/" + requestPath)
	if cleanPath == "/" {
		return ""
	}
	fullPath := filepath.Join(root, filepath.FromSlash(strings.TrimPrefix(cleanPath, "/")))

	file, err := os.Open(fullPath)
	if err != nil {
		return ""
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil || info.IsDir() {
		return ""
	}

	var header [512]byte
	n, _ := file.Read(header[:])
	if n <= 0 {
		return ""
	}

	return sniffVideoContentType(header[:n])
}

func sniffVideoContentType(header []byte) string {
	if len(header) >= 12 && bytes.Equal(header[4:8], []byte("ftyp")) {
		if string(header[8:12]) == "qt  " {
			return "video/quicktime"
		}
		return "video/mp4"
	}

	if len(header) >= 4 && bytes.Equal(header[:4], []byte{0x1a, 0x45, 0xdf, 0xa3}) {
		if bytes.Contains(header, []byte("webm")) {
			return "video/webm"
		}
		if bytes.Contains(header, []byte("matroska")) {
			return "video/x-matroska"
		}
	}

	return ""
}
