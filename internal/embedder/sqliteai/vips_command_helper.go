package sqliteai

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"
)

const (
	vipsCommandHelperEnv = "IMGSEARCH_SQLITEAI_VIPS_COMMAND_HELPER"
	vipsCommandHelperArg = "--imgsearch-sqliteai-vips-command-helper"
)

type vipsCommandHelperRequest struct {
	Name             string
	Args             []string
	DeadlineUnixNano int64
}

type vipsCommandHelperResponse struct {
	Output []byte
	Err    string
}

// init is used so the helper works when the current executable is either the
// main app binary or a `go test` binary.
func init() {
	if os.Getenv(vipsCommandHelperEnv) != "1" {
		return
	}
	for _, arg := range os.Args[1:] {
		if arg == vipsCommandHelperArg {
			runVipsCommandHelper()
			os.Exit(0)
		}
	}
}

func runVipsCommandHelper() {
	dec := gob.NewDecoder(os.Stdin)
	enc := gob.NewEncoder(os.Stdout)

	for {
		var req vipsCommandHelperRequest
		if err := dec.Decode(&req); err != nil {
			if errors.Is(err, io.EOF) {
				return
			}
			_, _ = fmt.Fprintf(os.Stderr, "imgsearch sqlite-ai vips helper: decode request: %v\n", err)
			return
		}

		ctx := context.Background()
		cancel := func() {}
		if req.DeadlineUnixNano != 0 {
			deadline := time.Unix(0, req.DeadlineUnixNano)
			ctx, cancel = context.WithDeadline(ctx, deadline)
		}

		cmd := exec.CommandContext(ctx, req.Name, req.Args...)
		cmd.Env = withoutEnvVar(os.Environ(), vipsCommandHelperEnv)

		out, err := cmd.CombinedOutput()
		cancel()
		errText := ""
		if err != nil {
			errText = err.Error()
		}

		if err := enc.Encode(vipsCommandHelperResponse{Output: out, Err: errText}); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "imgsearch sqlite-ai vips helper: encode response: %v\n", err)
			return
		}
	}
}

func withoutEnvVar(env []string, key string) []string {
	if key == "" {
		return env
	}

	out := make([]string, 0, len(env))
	prefix := key + "="
	for _, kv := range env {
		if len(kv) >= len(prefix) && kv[:len(prefix)] == prefix {
			continue
		}
		out = append(out, kv)
	}
	return out
}

type vipsCommandHelper struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	enc    *gob.Encoder
	dec    *gob.Decoder
	waitCh chan error
}

func startVipsCommandHelper(ctx context.Context) (*vipsCommandHelper, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("resolve current executable: %w", err)
	}
	cmd := exec.CommandContext(ctx, exe, vipsCommandHelperArg)
	cmd.Env = append(withoutEnvVar(os.Environ(), vipsCommandHelperEnv), vipsCommandHelperEnv+"=1")
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("open helper stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("open helper stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return nil, fmt.Errorf("start helper process: %w", err)
	}

	h := &vipsCommandHelper{
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
		enc:    gob.NewEncoder(stdin),
		dec:    gob.NewDecoder(stdout),
		waitCh: make(chan error, 1),
	}
	go func() {
		h.waitCh <- cmd.Wait()
	}()
	return h, nil
}

func (h *vipsCommandHelper) Close() error {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	stdin := h.stdin
	h.stdin = nil
	h.mu.Unlock()

	if stdin != nil {
		_ = stdin.Close()
	}
	if h.waitCh != nil {
		return <-h.waitCh
	}
	return nil
}

func (h *vipsCommandHelper) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	if h == nil {
		return nil, fmt.Errorf("vips command helper is nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	deadline := int64(0)
	if dl, ok := ctx.Deadline(); ok {
		deadline = dl.UnixNano()
	}
	req := vipsCommandHelperRequest{Name: name, Args: append([]string(nil), args...), DeadlineUnixNano: deadline}

	h.mu.Lock()
	defer h.mu.Unlock()
	if h.stdin == nil {
		return nil, fmt.Errorf("vips command helper is closed")
	}
	if err := h.enc.Encode(req); err != nil {
		return nil, err
	}
	var resp vipsCommandHelperResponse
	if err := h.dec.Decode(&resp); err != nil {
		return nil, err
	}
	if resp.Err != "" {
		return resp.Output, errors.New(resp.Err)
	}
	return resp.Output, nil
}
