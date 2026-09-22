package mediaops

import "testing"

func TestAnnotationState(t *testing.T) {
	cases := []struct {
		job       string
		hasText   bool
		requested bool
		want      string
	}{
		{"leased", true, false, AnnotationAnnotating},
		{"pending", true, false, AnnotationQueued},
		{"failed", false, false, AnnotationFailed},
		{"done", true, false, AnnotationDone},
		{"done", true, true, AnnotationQueued},
		{"", true, false, AnnotationDone},
		{"", false, false, AnnotationNone},
		{"", false, true, AnnotationQueued},
	}
	for _, c := range cases {
		if got := AnnotationState(c.job, c.hasText, c.requested); got != c.want {
			t.Fatalf("AnnotationState(%q,%v,%v)=%q want %q", c.job, c.hasText, c.requested, got, c.want)
		}
	}
}
