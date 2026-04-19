package live

import (
	"context"
	"database/sql"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"

	"imgsearch/internal/httputil"
	"imgsearch/internal/images"
	"imgsearch/internal/stats"
)

const (
	defaultPushInterval = 2 * time.Second
	defaultImagesLimit  = 120
	wsReadLimitBytes    = 1024
	wsWriteTimeout      = 5 * time.Second
	wsPongTimeout       = 60 * time.Second
	wsPingInterval      = 30 * time.Second
)

type Handler struct {
	DB           *sql.DB
	ModelID      int64
	Interval     time.Duration
	ImagesLimit  int
	ImagesOffset int
}

type Snapshot struct {
	Type   string              `json:"type"`
	Images images.ListResponse `json:"images"`
	Stats  stats.Response      `json:"stats"`
	SentAt string              `json:"sent_at"`
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		origin := strings.TrimSpace(r.Header.Get("Origin"))
		if origin == "" {
			return isRequestHostWellFormed(r.Host)
		}
		u, err := url.Parse(origin)
		if err != nil {
			return false
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return false
		}
		originHost, ok := normalizeHostPort(u.Host, u.Scheme)
		if !ok {
			return false
		}
		requestHost, ok := normalizeHostPort(r.Host, u.Scheme)
		if !ok {
			return false
		}
		return originHost == requestHost
	},
}

func NewHandler(h *Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, http.StatusInternalServerError, "live updates unavailable")
			return
		}
		if r.Method != http.MethodGet {
			httputil.WriteJSONError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		conn.SetReadLimit(wsReadLimitBytes)
		_ = conn.SetReadDeadline(time.Now().Add(wsPongTimeout))
		conn.SetPongHandler(func(string) error {
			return conn.SetReadDeadline(time.Now().Add(wsPongTimeout))
		})

		pushInterval := h.Interval
		if pushInterval <= 0 {
			pushInterval = defaultPushInterval
		}
		imagesLimit := h.ImagesLimit
		if imagesLimit <= 0 {
			imagesLimit = defaultImagesLimit
		}
		imagesOffset := h.ImagesOffset
		if imagesOffset < 0 {
			imagesOffset = 0
		}
		includeNSFW := parseIncludeNSFW(r)

		closed := make(chan struct{})
		go func() {
			defer close(closed)
			for {
				if _, _, err := conn.NextReader(); err != nil {
					return
				}
			}
		}()

		go func() {
			ticker := time.NewTicker(wsPingInterval)
			defer ticker.Stop()
			for {
				select {
				case <-r.Context().Done():
					return
				case <-closed:
					return
				case <-ticker.C:
					if err := writeControl(conn, websocket.PingMessage); err != nil {
						return
					}
				}
			}
		}()

		if err := writeSnapshot(r.Context(), conn, h.DB, h.ModelID, imagesLimit, imagesOffset, includeNSFW); err != nil {
			return
		}

		ticker := time.NewTicker(pushInterval)
		defer ticker.Stop()

		for {
			select {
			case <-r.Context().Done():
				return
			case <-closed:
				return
			case <-ticker.C:
				if err := writeSnapshot(r.Context(), conn, h.DB, h.ModelID, imagesLimit, imagesOffset, includeNSFW); err != nil {
					return
				}
			}
		}
	})
}

func writeSnapshot(ctx context.Context, conn *websocket.Conn, db *sql.DB, modelID int64, limit int, offset int, includeNSFW bool) error {
	imagesResp, err := images.List(ctx, db, modelID, limit, offset, includeNSFW)
	if err != nil {
		return err
	}
	statsResp, err := stats.Collect(ctx, db, modelID)
	if err != nil {
		return err
	}

	return writeWS(conn, Snapshot{
		Type:   "snapshot",
		Images: imagesResp,
		Stats:  statsResp,
		SentAt: time.Now().UTC().Format(time.RFC3339Nano),
	})
}

func writeWS(conn *websocket.Conn, payload any) error {
	if err := conn.SetWriteDeadline(time.Now().Add(wsWriteTimeout)); err != nil {
		return err
	}
	return conn.WriteJSON(payload)
}

func writeControl(conn *websocket.Conn, messageType int) error {
	return conn.WriteControl(messageType, nil, time.Now().Add(wsWriteTimeout))
}

func isRequestHostWellFormed(host string) bool {
	_, ok := normalizeHostPort(host, "http")
	return ok
}

func normalizeHostPort(host string, scheme string) (string, bool) {
	host = strings.TrimSpace(host)
	if host == "" {
		return "", false
	}

	parsed, err := url.Parse("//" + host)
	if err != nil {
		return "", false
	}
	hostname := strings.ToLower(strings.TrimSpace(parsed.Hostname()))
	if hostname == "" {
		return "", false
	}
	port := parsed.Port()
	if port == "" {
		port = defaultPortForScheme(scheme)
	}
	if port == "" {
		return "", false
	}

	return net.JoinHostPort(hostname, port), true
}

func defaultPortForScheme(scheme string) string {
	switch strings.ToLower(strings.TrimSpace(scheme)) {
	case "https", "wss":
		return "443"
	default:
		return "80"
	}
}

func parseIncludeNSFW(r *http.Request) bool {
	v := strings.TrimSpace(r.URL.Query().Get("include_nsfw"))
	if v == "" {
		return false
	}
	parsed, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return parsed
}
