package live

import (
	"context"
	"database/sql"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"imgsearch/internal/httputil"
	"imgsearch/internal/images"
	"imgsearch/internal/stats"
	"imgsearch/internal/videos"
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

	hubOnce sync.Once
	hub     *hub
}

type Snapshot struct {
	Type   string              `json:"type"`
	Images images.ListResponse `json:"images"`
	Videos videos.ListResponse `json:"videos"`
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
		if r.Method != http.MethodGet {
			httputil.WriteMethodNotAllowed(w, http.MethodGet)
			return
		}
		if h == nil || h.DB == nil {
			httputil.WriteJSONError(w, http.StatusServiceUnavailable, "live updates unavailable")
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
		includeNSFW := httputil.ParseIncludeNSFWQuery(r)
		h.hubOnce.Do(func() {
			h.hub = newHub(h.DB, h.ModelID, pushInterval, imagesLimit, imagesOffset)
		})

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

		sub := h.hub.subscribe(includeNSFW)
		defer h.hub.unsubscribe(sub)

		initial, _, err := h.hub.snapshot(r.Context(), includeNSFW)
		if err != nil {
			return
		}
		if err := writeWS(conn, initial); err != nil {
			return
		}

		for {
			select {
			case <-r.Context().Done():
				return
			case <-closed:
				return
			case snapshot := <-sub.ch:
				if err := writeWS(conn, snapshot); err != nil {
					return
				}
			}
		}
	})
}

// hub computes at most one snapshot per push interval per NSFW variant and
// fans it out to every connected client, and skips the computation entirely
// while the database fingerprint has not moved. Without it, N open tabs cost
// N full recomputations every interval on the single SQLite connection.
type hub struct {
	db       *sql.DB
	modelID  int64
	interval time.Duration
	limit    int
	offset   int

	mu          sync.Mutex
	subs        map[*subscriber]struct{}
	cached      map[bool]*Snapshot
	fingerprint changeFingerprint
	running     bool
	// computations counts full snapshot builds; tests use it to prove
	// idle intervals are free.
	computations int64
}

type subscriber struct {
	includeNSFW bool
	ch          chan *Snapshot
}

// changeFingerprint moves on any committed write: total_changes() covers
// writes on this connection (the in-process worker shares it), and
// data_version covers writes from any other connection or process.
type changeFingerprint struct {
	known        bool
	totalChanges int64
	dataVersion  int64
}

func newHub(db *sql.DB, modelID int64, interval time.Duration, limit int, offset int) *hub {
	return &hub{
		db:       db,
		modelID:  modelID,
		interval: interval,
		limit:    limit,
		offset:   offset,
		subs:     map[*subscriber]struct{}{},
		cached:   map[bool]*Snapshot{},
	}
}

func (h *hub) subscribe(includeNSFW bool) *subscriber {
	sub := &subscriber{includeNSFW: includeNSFW, ch: make(chan *Snapshot, 1)}
	h.mu.Lock()
	h.subs[sub] = struct{}{}
	if !h.running {
		h.running = true
		go h.loop()
	}
	h.mu.Unlock()
	return sub
}

func (h *hub) unsubscribe(sub *subscriber) {
	h.mu.Lock()
	delete(h.subs, sub)
	h.mu.Unlock()
}

// loop ticks while there are subscribers and broadcasts fresh snapshots.
func (h *hub) loop() {
	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()
	for range ticker.C {
		h.mu.Lock()
		if len(h.subs) == 0 {
			h.running = false
			h.mu.Unlock()
			return
		}
		variants := map[bool][]*subscriber{}
		for sub := range h.subs {
			variants[sub.includeNSFW] = append(variants[sub.includeNSFW], sub)
		}
		h.mu.Unlock()

		for includeNSFW, subs := range variants {
			snapshot, fresh, err := h.snapshot(context.Background(), includeNSFW)
			if err != nil || !fresh {
				continue
			}
			for _, sub := range subs {
				sub.deliver(snapshot)
			}
		}
	}
}

// deliver hands the newest snapshot to a client, replacing an unread one.
func (s *subscriber) deliver(snapshot *Snapshot) {
	select {
	case s.ch <- snapshot:
	default:
		select {
		case <-s.ch:
		default:
		}
		s.ch <- snapshot
	}
}

// snapshot returns the current snapshot for a variant. fresh reports whether
// it was rebuilt because the database changed (or was never built).
func (h *hub) snapshot(ctx context.Context, includeNSFW bool) (*Snapshot, bool, error) {
	fp, err := h.readFingerprint(ctx)
	if err != nil {
		return nil, false, err
	}

	h.mu.Lock()
	if fp != h.fingerprint {
		h.fingerprint = fp
		h.cached = map[bool]*Snapshot{}
	}
	if cached, ok := h.cached[includeNSFW]; ok {
		h.mu.Unlock()
		return cached, false, nil
	}
	h.mu.Unlock()

	built, err := h.build(ctx, includeNSFW)
	if err != nil {
		return nil, false, err
	}
	h.mu.Lock()
	h.cached[includeNSFW] = built
	h.computations++
	h.mu.Unlock()
	return built, true, nil
}

func (h *hub) readFingerprint(ctx context.Context) (changeFingerprint, error) {
	fp := changeFingerprint{known: true}
	if err := h.db.QueryRowContext(ctx, `SELECT total_changes(), (SELECT data_version FROM pragma_data_version)`).Scan(&fp.totalChanges, &fp.dataVersion); err != nil {
		return changeFingerprint{}, fmt.Errorf("live change fingerprint: %w", err)
	}
	return fp, nil
}

func (h *hub) build(ctx context.Context, includeNSFW bool) (*Snapshot, error) {
	imagesResp, err := images.List(ctx, h.db, h.modelID, h.limit, h.offset, includeNSFW)
	if err != nil {
		return nil, err
	}
	videosResp, err := videos.List(ctx, h.db, h.modelID, h.limit, h.offset, includeNSFW)
	if err != nil {
		return nil, err
	}
	statsResp, err := stats.Collect(ctx, h.db, h.modelID)
	if err != nil {
		return nil, err
	}
	return &Snapshot{
		Type:   "snapshot",
		Images: imagesResp,
		Videos: videosResp,
		Stats:  statsResp,
		SentAt: time.Now().UTC().Format(time.RFC3339Nano),
	}, nil
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
