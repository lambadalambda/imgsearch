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

func ToggleTagJSON(raw string, target string) (string, bool, error) {
	tags, err := DecodeJSON(raw)
	if err != nil {
		return "", false, err
	}
	updatedTags, isPresent := ToggleTag(tags, target)
	encodedTags, err := json.Marshal(updatedTags)
	if err != nil {
		return "", false, err
	}
	return string(encodedTags), isPresent, nil
}

// Normalize trims tags, drops empties, and removes case-insensitive
// duplicates while keeping the first spelling and order.
func Normalize(tags []string) []string {
	out := make([]string, 0, len(tags))
	seen := make(map[string]struct{}, len(tags))
	for _, tag := range tags {
		trimmed := strings.TrimSpace(tag)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if _, dup := seen[key]; dup {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, trimmed)
	}
	return out
}

// EncodeJSON renders tags as the JSON array stored in *_tags_json columns.
// nil encodes as "[]".
func EncodeJSON(tags []string) string {
	if tags == nil {
		tags = []string{}
	}
	encoded, err := json.Marshal(tags)
	if err != nil {
		return "[]"
	}
	return string(encoded)
}

// Merge computes the served tag list: the annotator's tags minus the ones
// the user removed, plus the user's own additions.
func Merge(annotator []string, user []string, removed []string) []string {
	drop := lowerSet(removed)
	merged := make([]string, 0, len(annotator)+len(user))
	for _, tag := range Normalize(annotator) {
		if _, gone := drop[strings.ToLower(tag)]; gone {
			continue
		}
		merged = append(merged, tag)
	}
	return Normalize(append(merged, user...))
}

// Diff splits a user-edited served list against the annotator's tags into
// the user's additions and the annotator tags the user removed.
func Diff(annotator []string, served []string) (user []string, removed []string) {
	annotatorSet := lowerSet(annotator)
	servedSet := lowerSet(served)
	user = []string{}
	for _, tag := range Normalize(served) {
		if _, fromAnnotator := annotatorSet[strings.ToLower(tag)]; !fromAnnotator {
			user = append(user, tag)
		}
	}
	removed = []string{}
	for _, tag := range Normalize(annotator) {
		if _, kept := servedSet[strings.ToLower(tag)]; !kept {
			removed = append(removed, tag)
		}
	}
	return user, removed
}

func lowerSet(tags []string) map[string]struct{} {
	set := make(map[string]struct{}, len(tags))
	for _, tag := range tags {
		trimmed := strings.TrimSpace(tag)
		if trimmed == "" {
			continue
		}
		set[strings.ToLower(trimmed)] = struct{}{}
	}
	return set
}
