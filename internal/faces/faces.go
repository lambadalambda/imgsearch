// Package faces implements local face detection and identity embedding.
//
// The pipeline mirrors InsightFace: an SCRFD detector finds faces and
// five-point landmarks, each face is aligned to the ArcFace template with a
// similarity transform, and an ArcFace-class recognition model produces a
// normalized identity vector. All preprocessing is pure Go so this package
// stays testable without native libraries; only engine.go touches ONNX
// Runtime (kept in a separate file for the same reason).
package faces

import "errors"

// ErrUnsupportedFormat reports image formats with no pure-Go decoder (AVIF).
// Callers should skip the file instead of failing the job.
var ErrUnsupportedFormat = errors.New("unsupported image format for face detection")

const (
	// DetInputSize is the square letterbox the detector runs at.
	DetInputSize = 640
	// DetThreshold filters raw SCRFD scores before NMS.
	DetThreshold = 0.5
	// DetNMSThreshold is the IoU cutoff for non-maximum suppression.
	DetNMSThreshold = 0.4
	// RecInputSize is the aligned face crop the recognition model expects.
	RecInputSize = 112
	// DetStrides and DetAnchors describe the 9-output SCRFD-10G layout:
	// score/bbox/landmark heads at strides 8/16/32 with 2 anchors each.
	DetAnchorCount = 2
)

// DetStrides are the feature strides of the SCRFD-10G detection heads.
var DetStrides = [3]int{8, 16, 32}

// ArcFaceTemplate is the canonical 112x112 five-point landmark template
// (left eye, right eye, nose, left mouth, right mouth).
var ArcFaceTemplate = [5][2]float64{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}
