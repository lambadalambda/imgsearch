package webui

import (
	"embed"
	"io/fs"
	"net/http"
	"path/filepath"
)

//go:embed static/*
var staticFiles embed.FS

func NewHandler(dataDir string) http.Handler {
	if dataDir == "" {
		dataDir = "."
	}

	assets, err := fs.Sub(staticFiles, "static")
	if err != nil {
		panic("webui assets unavailable: " + err.Error())
	}

	indexHTML, err := fs.ReadFile(assets, "index.html")
	if err != nil {
		panic("webui index unavailable: " + err.Error())
	}

	imagesRoot := filepath.Join(dataDir, "images")
	videosRoot := filepath.Join(dataDir, "videos")
	mediaMux := http.NewServeMux()
	mediaMux.Handle("/images/", http.StripPrefix("/images/", http.FileServer(http.Dir(imagesRoot))))
	mediaMux.Handle("/videos/", http.StripPrefix("/videos/", http.FileServer(http.Dir(videosRoot))))

	mux := http.NewServeMux()
	mux.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.FS(assets))))
	mux.Handle("/media/", http.StripPrefix("/media", mediaMux))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" && r.URL.Path != "/index.html" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(indexHTML)
	})

	return mux
}
