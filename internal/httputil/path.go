package httputil

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func ParseItemIDPath(path string, prefix string) (int64, error) {
	idText := strings.TrimPrefix(path, prefix)
	if idText == path || idText == "" || strings.ContainsRune(idText, '/') {
		return 0, fmt.Errorf("invalid id path")
	}
	return strconv.ParseInt(idText, 10, 64)
}

func ParseItemActionIDPath(path string, prefix string, action string) (int64, error) {
	suffix := "/" + strings.Trim(action, "/")
	if suffix == "/" || !strings.HasSuffix(path, suffix) {
		return 0, fmt.Errorf("invalid action path")
	}
	return ParseItemIDPath(strings.TrimSuffix(path, suffix), prefix)
}

func BoolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func RemoveStoredPath(dataDir string, rel string) error {
	trimmedDataDir := strings.TrimSpace(dataDir)
	trimmedRel := strings.TrimSpace(rel)
	if trimmedDataDir == "" || trimmedRel == "" {
		return nil
	}

	baseAbs, err := filepath.Abs(trimmedDataDir)
	if err != nil {
		return fmt.Errorf("resolve data dir: %w", err)
	}
	targetAbs := filepath.Clean(filepath.Join(baseAbs, filepath.FromSlash(trimmedRel)))

	baseCheckPath, err := resolvePathForContainment(baseAbs)
	if err != nil {
		return fmt.Errorf("resolve data dir symlinks: %w", err)
	}
	targetCheckPath, err := resolvePathForContainment(targetAbs)
	if err != nil {
		return fmt.Errorf("resolve stored path: %w", err)
	}

	relToBase, err := filepath.Rel(baseCheckPath, targetCheckPath)
	if err != nil {
		return fmt.Errorf("resolve stored path: %w", err)
	}
	if relToBase == "." || relToBase == ".." || strings.HasPrefix(relToBase, ".."+string(os.PathSeparator)) {
		return fmt.Errorf("stored path escapes data dir")
	}

	if err := os.Remove(targetAbs); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func resolvePathForContainment(path string) (string, error) {
	cleaned := filepath.Clean(path)
	probe := cleaned
	missingSuffix := make([]string, 0, 4)

	for {
		resolved, err := filepath.EvalSymlinks(probe)
		if err == nil {
			for i := len(missingSuffix) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, missingSuffix[i])
			}
			return resolved, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}

		next := filepath.Dir(probe)
		if next == probe {
			return cleaned, nil
		}
		missingSuffix = append(missingSuffix, filepath.Base(probe))
		probe = next
	}
}
