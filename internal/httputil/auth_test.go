package httputil

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAPIAuthMiddlewareAllowsNonAPIRequests(t *testing.T) {
	h := NewAPIAuthMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusNoContent)
	}
}

func TestAPIAuthMiddlewareRejectsUnauthorizedAPIRequests(t *testing.T) {
	h := NewAPIAuthMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusUnauthorized, rr.Body.String())
	}
}

func TestAPIAuthMiddlewareAcceptsCookieToken(t *testing.T) {
	h := NewAPIAuthMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	req.AddCookie(&http.Cookie{Name: APIKeyCookieName, Value: apiKeyCookieValue("secret-token")})
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusNoContent)
	}
}

func TestAPIAuthMiddlewareAcceptsBearerToken(t *testing.T) {
	h := NewAPIAuthMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d want=%d body=%s", rr.Code, http.StatusNoContent, rr.Body.String())
	}
}

func TestAPIAuthMiddlewareRejectsInvalidToken(t *testing.T) {
	h := NewAPIAuthMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	req.Header.Set(APIKeyHeaderName, "wrong-token")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusUnauthorized)
	}
}

func TestAPIKeyCookieMiddlewareSetsCookieOnWebRequests(t *testing.T) {
	h := NewAPIKeyCookieMiddleware("secret-token")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	webReq := httptest.NewRequest(http.MethodGet, "/", nil)
	webRR := httptest.NewRecorder()
	h.ServeHTTP(webRR, webReq)
	if webRR.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d want=%d", webRR.Code, http.StatusNoContent)
	}
	if cookie := webRR.Header().Get("Set-Cookie"); cookie == "" {
		t.Fatalf("expected auth cookie on web request")
	}
	resp := webRR.Result()
	defer func() { _ = resp.Body.Close() }()
	cookies := resp.Cookies()
	if len(cookies) != 1 {
		t.Fatalf("cookie count: got=%d want=1", len(cookies))
	}
	if cookies[0].Name != APIKeyCookieName {
		t.Fatalf("cookie name: got=%q want=%q", cookies[0].Name, APIKeyCookieName)
	}
	if cookies[0].Value != apiKeyCookieValue("secret-token") {
		t.Fatalf("unexpected cookie value")
	}

	apiReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	apiRR := httptest.NewRecorder()
	h.ServeHTTP(apiRR, apiReq)
	if apiRR.Header().Get("Set-Cookie") != "" {
		t.Fatalf("did not expect auth cookie to be set on api request")
	}
}

func TestAPIAuthMiddlewareDisabledWithoutToken(t *testing.T) {
	h := NewAPIAuthMiddleware("")(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status: got=%d want=%d", rr.Code, http.StatusNoContent)
	}
}

// withAPISecurity mirrors the production middleware chain in
// `cmd/imgsearch.withAPISecurity` so chained tests exercise the real order:
// API auth first, then cookie mint on the response.
func withAPISecurity(apiKey string, next http.Handler) http.Handler {
	h := NewAPIAuthMiddleware(apiKey)(next)
	h = NewAPIKeyCookieMiddleware(apiKey)(h)
	return h
}

// TestAPISecurityChainMintsCookieOnWebAndAcceptsItOnAPI is the regression
// test for issue #054: anyone who can reach the UI is intentionally trusted
// to use the API. A non-API request mints the auth cookie, and a subsequent
// same-origin API call carrying that cookie is accepted.
func TestAPISecurityChainMintsCookieOnWebAndAcceptsItOnAPI(t *testing.T) {
	h := withAPISecurity("secret-token", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isAPIPath(r.URL.Path) {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))

	webReq := httptest.NewRequest(http.MethodGet, "/", nil)
	webRR := httptest.NewRecorder()
	h.ServeHTTP(webRR, webReq)
	if webRR.Code != http.StatusOK {
		t.Fatalf("web status: got=%d want=%d", webRR.Code, http.StatusOK)
	}
	webResp := webRR.Result()
	defer func() { _ = webResp.Body.Close() }()
	webCookies := webResp.Cookies()
	if len(webCookies) != 1 {
		t.Fatalf("expected one cookie on web response, got %d", len(webCookies))
	}
	if webCookies[0].Name != APIKeyCookieName {
		t.Fatalf("web cookie name: got=%q want=%q", webCookies[0].Name, APIKeyCookieName)
	}

	apiReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	apiReq.AddCookie(webCookies[0])
	apiRR := httptest.NewRecorder()
	h.ServeHTTP(apiRR, apiReq)
	if apiRR.Code != http.StatusOK {
		t.Fatalf("api status with cookie: got=%d want=%d body=%s", apiRR.Code, http.StatusOK, apiRR.Body.String())
	}
}

// TestAPISecurityChainAcceptsAPIHeaderAuthIndependentlyOfCookies ensures the
// API-header path (X-Imgsearch-API-Key / Authorization: Bearer) keeps working
// for scripts and importers that don't carry browser cookies.
func TestAPISecurityChainAcceptsAPIHeaderAuthIndependentlyOfCookies(t *testing.T) {
	h := withAPISecurity("secret-token", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	bearerReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	bearerReq.Header.Set("Authorization", "Bearer secret-token")
	bearerRR := httptest.NewRecorder()
	h.ServeHTTP(bearerRR, bearerReq)
	if bearerRR.Code != http.StatusOK {
		t.Fatalf("bearer status: got=%d want=%d body=%s", bearerRR.Code, http.StatusOK, bearerRR.Body.String())
	}

	headerReq := httptest.NewRequest(http.MethodGet, "/api/live", nil)
	headerReq.Header.Set(APIKeyHeaderName, "secret-token")
	headerRR := httptest.NewRecorder()
	h.ServeHTTP(headerRR, headerReq)
	if headerRR.Code != http.StatusOK {
		t.Fatalf("header status: got=%d want=%d body=%s", headerRR.Code, http.StatusOK, headerRR.Body.String())
	}
}
