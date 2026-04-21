package httputil

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"net/http"
	"strings"
)

const (
	APIKeyHeaderName = "X-Imgsearch-API-Key"
	APIKeyCookieName = "imgsearch_api_key"
	apiKeyCookieSalt = "imgsearch-api-cookie-v1"
)

func NewAPIAuthMiddleware(apiKey string) func(http.Handler) http.Handler {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return func(next http.Handler) http.Handler {
			return next
		}
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !isAPIPath(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}

			token := requestAPIKeyToken(r)
			if secureTokenEqual(token, apiKey) {
				next.ServeHTTP(w, r)
				return
			}

			cookieToken := requestAPIKeyCookieToken(r)
			if secureTokenEqual(cookieToken, apiKeyCookieValue(apiKey)) {
				next.ServeHTTP(w, r)
				return
			}
			WriteJSONError(w, http.StatusUnauthorized, "unauthorized")
		})
	}
}

func NewAPIKeyCookieMiddleware(apiKey string) func(http.Handler) http.Handler {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return func(next http.Handler) http.Handler {
			return next
		}
	}

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !isAPIPath(r.URL.Path) {
				cookieValue := apiKeyCookieValue(apiKey)
				http.SetCookie(w, &http.Cookie{
					Name:     APIKeyCookieName,
					Value:    cookieValue,
					Path:     "/",
					HttpOnly: true,
					SameSite: http.SameSiteStrictMode,
					Secure:   r.TLS != nil,
				})
			}
			next.ServeHTTP(w, r)
		})
	}
}

func requestAPIKeyToken(r *http.Request) string {
	authorization := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(authorization), "bearer ") {
		bearerToken := strings.TrimSpace(authorization[len("Bearer "):])
		if bearerToken != "" {
			return bearerToken
		}
	}

	headerToken := strings.TrimSpace(r.Header.Get(APIKeyHeaderName))
	if headerToken != "" {
		return headerToken
	}

	return ""
}

func requestAPIKeyCookieToken(r *http.Request) string {
	if cookie, err := r.Cookie(APIKeyCookieName); err == nil {
		return strings.TrimSpace(cookie.Value)
	}
	return ""
}

func apiKeyCookieValue(apiKey string) string {
	mac := hmac.New(sha256.New, []byte(apiKey))
	_, _ = mac.Write([]byte(apiKeyCookieSalt))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func secureTokenEqual(token string, expected string) bool {
	if strings.TrimSpace(token) == "" || strings.TrimSpace(expected) == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(token), []byte(expected)) == 1
}

func isAPIPath(path string) bool {
	return path == "/api" || strings.HasPrefix(path, "/api/")
}
