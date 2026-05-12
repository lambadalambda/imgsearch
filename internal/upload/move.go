package upload

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

type renameFunc func(string, string) error

func moveFile(src, dst string) error {
	return moveFileWithRename(src, dst, os.Rename)
}

func moveFileWithRename(src, dst string, rename renameFunc) error {
	if err := rename(src, dst); err == nil {
		return nil
	} else if !errors.Is(err, syscall.EXDEV) {
		return err
	}

	if err := copyFileIntoPlace(src, dst); err != nil {
		return err
	}
	if err := os.Remove(src); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("remove source after copy: %w", err)
	}
	return nil
}

func copyFileIntoPlace(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open source for copy: %w", err)
	}
	defer func() { _ = in.Close() }()

	info, err := in.Stat()
	if err != nil {
		return fmt.Errorf("stat source for copy: %w", err)
	}

	dstDir := filepath.Dir(dst)
	tmp, err := os.CreateTemp(dstDir, ".move-*")
	if err != nil {
		return fmt.Errorf("create destination temp file: %w", err)
	}
	tmpPath := tmp.Name()
	committed := false
	defer func() {
		_ = tmp.Close()
		if !committed {
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err := io.Copy(tmp, in); err != nil {
		return fmt.Errorf("copy file contents: %w", err)
	}
	if err := tmp.Chmod(info.Mode().Perm()); err != nil {
		return fmt.Errorf("set copied file mode: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close copied file: %w", err)
	}
	if err := os.Rename(tmpPath, dst); err != nil {
		return fmt.Errorf("move copied file into place: %w", err)
	}
	committed = true
	return nil
}
