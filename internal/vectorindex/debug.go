package vectorindex

import "context"

type SearchDebug struct {
	Backend   string
	Strategy  string
	Quantized bool
}

type searchDebugKey struct{}

func WithSearchDebug(ctx context.Context, target *SearchDebug) context.Context {
	if ctx == nil || target == nil {
		return ctx
	}
	return context.WithValue(ctx, searchDebugKey{}, target)
}

func SetSearchDebug(ctx context.Context, debug SearchDebug) {
	if ctx == nil {
		return
	}
	target, ok := ctx.Value(searchDebugKey{}).(*SearchDebug)
	if !ok || target == nil {
		return
	}
	*target = debug
}
