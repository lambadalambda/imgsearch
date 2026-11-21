package faces

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestEstimateSimilarityIdentity(t *testing.T) {
	src := [5][2]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}
	m, err := EstimateSimilarity(src, src)
	if err != nil {
		t.Fatal(err)
	}
	if !approx(m.Scale, 1, 1e-9) || !approx(m.R[0][0], 1, 1e-9) || !approx(m.R[1][1], 1, 1e-9) ||
		!approx(m.R[0][1], 0, 1e-9) || !approx(m.R[1][0], 0, 1e-9) ||
		!approx(m.T[0], 0, 1e-9) || !approx(m.T[1], 0, 1e-9) {
		t.Fatalf("%+v", m)
	}
}

func TestEstimateSimilarityScaleTranslate(t *testing.T) {
	src := [5][2]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}
	var dst [5][2]float64
	for i, p := range src {
		dst[i] = [2]float64{3*p[0] + 10, 3*p[1] - 4}
	}
	m, err := EstimateSimilarity(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if !approx(m.Scale, 3, 1e-9) || !approx(m.T[0], 10, 1e-9) || !approx(m.T[1], -4, 1e-9) {
		t.Fatalf("%+v", m)
	}
}

func TestEstimateSimilarityRotation(t *testing.T) {
	src := [5][2]float64{{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {0.5, 0.5}}
	var dst [5][2]float64 // 90-degree rotation: (x, y) -> (-y, x)
	for i, p := range src {
		dst[i] = [2]float64{-p[1], p[0]}
	}
	m, err := EstimateSimilarity(src, dst)
	if err != nil {
		t.Fatal(err)
	}
	if !approx(m.Scale, 1, 1e-6) || !approx(m.R[0][0], 0, 1e-6) || !approx(m.R[0][1], -1, 1e-6) ||
		!approx(m.R[1][0], 1, 1e-6) || !approx(m.R[1][1], 0, 1e-6) {
		t.Fatalf("%+v", m)
	}
	// Round-trip every point through the fitted transform.
	for i, p := range src {
		qx := m.Scale*(m.R[0][0]*p[0]+m.R[0][1]*p[1]) + m.T[0]
		qy := m.Scale*(m.R[1][0]*p[0]+m.R[1][1]*p[1]) + m.T[1]
		if !approx(qx, dst[i][0], 1e-6) || !approx(qy, dst[i][1], 1e-6) {
			t.Fatalf("point %d maps to (%v,%v), want %v", i, qx, qy, dst[i])
		}
	}
}

func TestEstimateSimilarityDegenerate(t *testing.T) {
	var src, dst [5][2]float64
	for i := range src {
		src[i] = [2]float64{1, 1}
		dst[i] = [2]float64{2, 2}
	}
	if _, err := EstimateSimilarity(src, dst); err == nil {
		t.Fatal("identical points must fail")
	}
}

func TestAnchorCentersMatchReference(t *testing.T) {
	// Stride 32 on a 640 canvas: 20x20 grid, 2 anchors each, centers at
	// multiples of 32 starting from 0 (top-left of each cell).
	c := anchorCenters(20, 20, 32)
	if len(c) != 800 {
		t.Fatalf("len = %d", len(c))
	}
	if c[0] != [2]float64{0, 0} || c[1] != [2]float64{0, 0} || c[2] != [2]float64{32, 0} {
		t.Fatalf("%v %v %v", c[0], c[1], c[2])
	}
	if last := c[799]; last != [2]float64{19 * 32, 19 * 32} {
		t.Fatalf("last = %v", last)
	}
}

func TestDecodeDetectionsDistances(t *testing.T) {
	stride := 8
	levels := DetInputSize / stride // 80
	n := levels * levels * DetAnchorCount
	scores := make([]float32, n)
	bbox := make([]float32, n*4)
	kps := make([]float32, n*10)
	// Fire the anchor at grid cell (10, 5), first of its pair.
	idx := (5*levels + 10) * DetAnchorCount
	scores[idx] = 0.9
	// Distances in feature pixels: left/top/right/bottom = 2 cells each.
	bbox[idx*4] = 2
	bbox[idx*4+1] = 2
	bbox[idx*4+2] = 2
	bbox[idx*4+3] = 2
	// Landmark offsets: 1 cell right and down from the center.
	for j := 0; j < 5; j++ {
		kps[idx*10+j*2] = 1
		kps[idx*10+j*2+1] = 1
	}
	dets := DecodeDetections(scores, bbox, kps, DetInputSize, stride, DetThreshold)
	if len(dets) != 1 {
		t.Fatalf("dets = %d", len(dets))
	}
	d := dets[0]
	cx, cy := float64(10*stride), float64(5*stride)
	if !approx(d.X1, cx-16, 1e-6) || !approx(d.Y1, cy-16, 1e-6) || !approx(d.X2, cx+16, 1e-6) || !approx(d.Y2, cy+16, 1e-6) {
		t.Fatalf("%+v", d)
	}
	for j := 0; j < 5; j++ {
		if !approx(d.Landmarks[j][0], cx+8, 1e-6) || !approx(d.Landmarks[j][1], cy+8, 1e-6) {
			t.Fatalf("landmark %d = %v", j, d.Landmarks[j])
		}
	}
}

func TestNMSKeepsHighestAndDropsOverlap(t *testing.T) {
	a := RawDetection{X1: 0, Y1: 0, X2: 100, Y2: 100, Score: 0.9}
	b := RawDetection{X1: 5, Y1: 5, X2: 100, Y2: 100, Score: 0.8} // near-duplicate
	c := RawDetection{X1: 200, Y1: 200, X2: 300, Y2: 300, Score: 0.7}
	kept := NMS([]RawDetection{b, c, a}, DetNMSThreshold) // shuffled input
	if len(kept) != 2 || kept[0].Score != 0.9 || kept[1].Score != 0.7 {
		t.Fatalf("%+v", kept)
	}
}

func TestAlignFaceBlobShapeAndChannels(t *testing.T) {
	// Synthetic face: red left half, blue right half with template landmarks.
	img := image.NewRGBA(image.Rect(0, 0, 200, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 200; x++ {
			if x < 100 {
				img.Set(x, y, color.RGBA{255, 0, 0, 255})
			} else {
				img.Set(x, y, color.RGBA{0, 0, 255, 255})
			}
		}
	}
	// Identity-ish landmarks already in template space scaled to 200px.
	var lm [5][2]float64
	for i, p := range ArcFaceTemplate {
		lm[i] = [2]float64{p[0] / 112 * 200, p[1] / 112 * 200}
	}
	blob, err := AlignFace(img, lm)
	if err != nil {
		t.Fatal(err)
	}
	if len(blob) != 3*RecInputSize*RecInputSize {
		t.Fatalf("len = %d", len(blob))
	}
	// Left eye region samples the red half: B channel low, R channel high.
	// Blob layout is BGR planes; left-eye template point is (38, 51).
	at := func(c, x, y int) float32 { return blob[c*RecInputSize*RecInputSize+y*RecInputSize+x] }
	b, r := at(0, 38, 51), at(2, 38, 51)
	if !(b < -0.5 && r > 0.5) {
		t.Fatalf("left eye pixel B=%v R=%v, want blue-low red-high", b, r)
	}
}
