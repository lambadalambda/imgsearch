package httputil

import (
	"net/http"
	"strconv"
	"strings"
)

const maxLimitQueryValue = 200

// ParseLimitQuery parses the "limit" query parameter.
// Invalid or out-of-range values fall back to the provided default.
func ParseLimitQuery(r *http.Request, fallback int) int {
	v := r.URL.Query().Get("limit")
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 || n > maxLimitQueryValue {
		return fallback
	}
	return n
}

// ParseOffsetQuery parses the "offset" query parameter.
// Invalid values return 0. When max > 0, values above max also return 0.
// Pass max <= 0 to disable the upper bound check.
func ParseOffsetQuery(r *http.Request, max int) int {
	v := r.URL.Query().Get("offset")
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0
	}
	if max > 0 && n > max {
		return 0
	}
	return n
}

// ParseIncludeNSFWQuery parses include_nsfw as a boolean query parameter.
// Invalid or missing values default to false.
func ParseIncludeNSFWQuery(r *http.Request) bool {
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

// ParseOrderQuery parses the "order" query parameter against a small allow-list.
// Missing or unknown values fall back to the provided default.
func ParseOrderQuery(r *http.Request, fallback string, allowed ...string) string {
	v := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("order")))
	if v == "" {
		return fallback
	}
	for _, candidate := range allowed {
		if v == strings.ToLower(candidate) {
			return v
		}
	}
	return fallback
}

// ParseInt64Query parses a named int64 query parameter.
// Missing or invalid values fall back to the provided default.
func ParseInt64Query(r *http.Request, name string, fallback int64) int64 {
	v := strings.TrimSpace(r.URL.Query().Get(name))
	if v == "" {
		return fallback
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return fallback
	}
	return n
}
