package mediaops

import "strings"

// Annotation states exposed per item so the UI can show progress.
const (
	AnnotationQueued     = "queued"     // an annotation job is waiting
	AnnotationAnnotating = "annotating" // a worker holds the job lease
	AnnotationFailed     = "failed"     // the job exhausted its attempts
	AnnotationDone       = "done"       // text is present and no job is active
	AnnotationNone       = "none"       // no text and no job
)

// AnnotationState derives the user-facing state from the annotation job's
// state ("" when no job exists), whether annotation text is present, and the
// re-annotation flag.
func AnnotationState(jobState string, hasText bool, reannotateRequested bool) string {
	switch strings.ToLower(strings.TrimSpace(jobState)) {
	case "leased":
		return AnnotationAnnotating
	case "pending":
		return AnnotationQueued
	case "failed":
		return AnnotationFailed
	}
	if reannotateRequested {
		return AnnotationQueued
	}
	if hasText {
		return AnnotationDone
	}
	return AnnotationNone
}
