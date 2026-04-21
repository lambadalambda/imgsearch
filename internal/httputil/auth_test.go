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
