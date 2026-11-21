package faces

import (
	"sort"
)

// RawDetection is one decoded face in original-image coordinates.
type RawDetection struct {
	X1, Y1, X2, Y2 float64
	Score          float64
	Landmarks      [5][2]float64
}

// anchorCenters returns grid centers (top-left of each cell, not cell
// centers, matching the SCRFD reference) scaled by stride, with each center
// repeated DetAnchorCount times.
func anchorCenters(height, width, stride int) [][2]float64 {
	out := make([][2]float64, 0, height*width*DetAnchorCount)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			for a := 0; a < DetAnchorCount; a++ {
				out = append(out, [2]float64{float64(x * stride), float64(y * stride)})
			}
		}
	}
	return out
}

// DecodeDetections converts one stride level of SCRFD outputs to boxes and
// landmarks in letterboxed-canvas coordinates. bbox and kps are decoded as
// distances in feature pixels, so they are scaled by stride first.
func DecodeDetections(scores, bbox, kps []float32, inputSize, stride int, threshold float64) []RawDetection {
	levels := inputSize / stride
	centers := anchorCenters(levels, levels, stride)
	n := len(centers)
	if len(scores) < n || len(bbox) < n*4 || len(kps) < n*10 {
		return nil
	}
	var out []RawDetection
	for i := 0; i < n; i++ {
		if float64(scores[i]) < threshold {
			continue
		}
		cx, cy := centers[i][0], centers[i][1]
		b := bbox[i*4 : i*4+4]
		x1 := cx - float64(b[0])*float64(stride)
		y1 := cy - float64(b[1])*float64(stride)
		x2 := cx + float64(b[2])*float64(stride)
		y2 := cy + float64(b[3])*float64(stride)
		det := RawDetection{X1: x1, Y1: y1, X2: x2, Y2: y2, Score: float64(scores[i])}
		k := kps[i*10 : i*10+10]
		for j := 0; j < 5; j++ {
			dx := float64(k[j*2]) * float64(stride)
			dy := float64(k[j*2+1]) * float64(stride)
			// The reference alternates points[:, 0] for x and points[:, 1]
			// for y; centers are (x, y) so even slots offset x, odd slots y.
			det.Landmarks[j][0] = cx + dx
			det.Landmarks[j][1] = cy + dy
		}
		out = append(out, det)
	}
	return out
}

// ToOriginal rescales canvas coordinates back to the source image.
func (d RawDetection) ToOriginal(scale float64, origW, origH int) RawDetection {
	clamp := func(v, lo, hi float64) float64 {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
	d.X1 = clamp(d.X1/scale, 0, float64(origW))
	d.Y1 = clamp(d.Y1/scale, 0, float64(origH))
	d.X2 = clamp(d.X2/scale, 0, float64(origW))
	d.Y2 = clamp(d.Y2/scale, 0, float64(origH))
	for j := range d.Landmarks {
		d.Landmarks[j][0] = clamp(d.Landmarks[j][0]/scale, 0, float64(origW))
		d.Landmarks[j][1] = clamp(d.Landmarks[j][1]/scale, 0, float64(origH))
	}
	return d
}

// NMS keeps the highest-scoring box per overlapping group (IoU with the
// +1 pixel convention, matching the SCRFD reference). Input order is
// ignored; output is score-descending and stable for ties.
func NMS(dets []RawDetection, threshold float64) []RawDetection {
	order := make([]int, len(dets))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return dets[order[a]].Score > dets[order[b]].Score })
	areas := make([]float64, len(dets))
	for i, d := range dets {
		areas[i] = (d.X2 - d.X1 + 1) * (d.Y2 - d.Y1 + 1)
	}
	var keep []RawDetection
	active := make([]bool, len(dets))
	for i := range active {
		active[i] = true
	}
	for _, idx := range order {
		if !active[idx] {
			continue
		}
		keep = append(keep, dets[idx])
		for _, other := range order {
			if other == idx || !active[other] {
				continue
			}
			xx1 := max(dets[idx].X1, dets[other].X1)
			yy1 := max(dets[idx].Y1, dets[other].Y1)
			xx2 := min(dets[idx].X2, dets[other].X2)
			yy2 := min(dets[idx].Y2, dets[other].Y2)
			w := max(0, xx2-xx1+1)
			h := max(0, yy2-yy1+1)
			inter := w * h
			if inter/(areas[idx]+areas[other]-inter) > threshold {
				active[other] = false
			}
		}
	}
	return keep
}
