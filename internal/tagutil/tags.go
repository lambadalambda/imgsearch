package tagutil

import (
	"encoding/json"
	"strings"
)

func DecodeJSON(raw string) ([]string, error) {
	if raw == "" {
		return nil, nil
	}
	var tags []string
	if err := json.Unmarshal([]byte(raw), &tags); err != nil {
		return nil, err
	}
	return tags, nil
}

func ToggleTag(tags []string, target string) ([]string, bool) {
	normalizedTarget := strings.ToLower(strings.TrimSpace(target))
	if normalizedTarget == "" {
		cloned := make([]string, 0, len(tags))
		for _, tag := range tags {
			trimmed := strings.TrimSpace(tag)
			if trimmed == "" {
				continue
			}
			cloned = append(cloned, trimmed)
		}
		return cloned, false
	}

	updated := make([]string, 0, len(tags)+1)
	hadTag := false
	for _, tag := range tags {
		trimmed := strings.TrimSpace(tag)
		if trimmed == "" {
			continue
		}
		if strings.EqualFold(trimmed, normalizedTarget) {
			hadTag = true
			continue
		}
		updated = append(updated, trimmed)
	}

	if hadTag {
		return updated, false
	}
	updated = append(updated, normalizedTarget)
	return updated, true
}
