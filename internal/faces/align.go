package faces

import (
	"fmt"
	"image"
	"math"
)

// Similarity is a 2D similarity transform mapping source points to
// destination points: dst = scale * R * src + t.
type Similarity struct {
	R     [2][2]float64
	T     [2]float64
	Scale float64
}

// EstimateSimilarity fits a similarity transform from src to dst with
// Umeyama's method (closed-form 2x2 SVD). It rejects degenerate inputs.
func EstimateSimilarity(src, dst [5][2]float64) (Similarity, error) {
	var scx, scy, dcx, dcy float64
	for i := 0; i < 5; i++ {
		scx += src[i][0]
		scy += src[i][1]
		dcx += dst[i][0]
		dcy += dst[i][1]
	}
	scx /= 5
	scy /= 5
	dcx /= 5
	dcy /= 5
	// H = src'^T * dst' (2x2).
	var h [2][2]float64
	var srcVar float64
	for i := 0; i < 5; i++ {
		sx, sy := src[i][0]-scx, src[i][1]-scy
		dx, dy := dst[i][0]-dcx, dst[i][1]-dcy
		h[0][0] += sx * dx
		h[0][1] += sx * dy
		h[1][0] += sy * dx
		h[1][1] += sy * dy
		srcVar += sx*sx + sy*sy
	}
	if srcVar == 0 {
		return Similarity{}, fmt.Errorf("degenerate source landmarks")
	}
	u, s, v := svd2(h)
	// VU^T with V/U row-major and eigenvectors as columns.
	m00 := v[0][0]*u[0][0] + v[0][1]*u[0][1]
	m01 := v[0][0]*u[1][0] + v[0][1]*u[1][1]
	m10 := v[1][0]*u[0][0] + v[1][1]*u[0][1]
	m11 := v[1][0]*u[1][0] + v[1][1]*u[1][1]
	// Reflection fix: R = V * diag(1, det(VU^T)) * U^T.
	sign := 1.0
	if m00*m11-m01*m10 < 0 {
		sign = -1
	}
	r00 := v[0][0]*u[0][0] + sign*v[0][1]*u[0][1]
	r01 := v[0][0]*u[1][0] + sign*v[0][1]*u[1][1]
	r10 := v[1][0]*u[0][0] + sign*v[1][1]*u[0][1]
	r11 := v[1][0]*u[1][0] + sign*v[1][1]*u[1][1]
	scale := (s[0] + sign*s[1]) / srcVar
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return Similarity{}, fmt.Errorf("degenerate similarity scale %v", scale)
	}
	return Similarity{
		R:     [2][2]float64{{r00, r01}, {r10, r11}},
		T:     [2]float64{dcx - scale*(r00*scx+r01*scy), dcy - scale*(r10*scx+r11*scy)},
		Scale: scale,
	}, nil
}

// svd2 returns U, singular values, V with H = U * diag(s) * V^T.
func svd2(h [2][2]float64) (u [2][2]float64, s [2]float64, v [2][2]float64) {
	a, b, c, d := h[0][0], h[0][1], h[1][0], h[1][1]
	// Eigen-decomposition of symmetric M = H^T H.
	m00 := a*a + c*c
	m01 := a*b + c*d
	m11 := b*b + d*d
	trace, det := m00+m11, m00*m11-m01*m01
	disc := math.Sqrt(max(0, trace*trace/4-det))
	l1, l2 := trace/2+disc, trace/2-disc
	s = [2]float64{math.Sqrt(max(0, l1)), math.Sqrt(max(0, l2))}
	// Eigenvectors of M give V (first column for l1).
	eigvec := func(l float64) [2]float64 {
		if math.Abs(m01) > 1e-12 {
			x := [2]float64{l - m11, m01}
			n := math.Hypot(x[0], x[1])
			if n > 0 {
				return [2]float64{x[0] / n, x[1] / n}
			}
		}
		if m00 >= m11 {
			return [2]float64{1, 0}
		}
		return [2]float64{0, 1}
	}
	v1 := eigvec(l1)
	v2 := eigvec(l2)
	// Equal eigenvalues (e.g. symmetric point sets) make the picker return
	// the same vector twice; the perpendicular completes an orthonormal pair
	// and is the second eigenvector whenever eigenvalues differ.
	if math.Abs(v1[0]*v2[0]+v1[1]*v2[1]) > 1-1e-9 {
		v2 = [2]float64{-v1[1], v1[0]}
	}
	// Row-major V with eigenvectors as columns: V[i][k] = eigvec(l_k)[i].
	v = [2][2]float64{{v1[0], v2[0]}, {v1[1], v2[1]}}
	// Columns of U: u_i = H v_i / s_i, with orthogonal fallback.
	for i := 0; i < 2; i++ {
		if s[i] > 1e-12 {
			u[0][i] = (a*v[0][i] + b*v[1][i]) / s[i]
			u[1][i] = (c*v[0][i] + d*v[1][i]) / s[i]
		}
	}
	if s[0] <= 1e-12 && s[1] <= 1e-12 {
		u = [2][2]float64{{1, 0}, {0, 1}}
	} else if s[1] <= 1e-12 {
		u[0][1], u[1][1] = -u[1][0], u[0][0]
	} else if s[0] <= 1e-12 {
		u[0][0], u[1][0] = -u[1][1], u[0][1]
	}
	return u, s, v
}

// AlignFace warps img so landmarks meet the ArcFace template and returns the
// recognition blob: NCHW BGR floats of (v - 127.5) / 127.5 at RecInputSize.
func AlignFace(img image.Image, landmarks [5][2]float64) ([]float32, error) {
	m, err := EstimateSimilarity(landmarks, ArcFaceTemplate)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	sample := func(x, y float64) (r, g, bl float64) {
		if x < 0 {
			x = 0
		}
		if y < 0 {
			y = 0
		}
		if x > float64(w-1) {
			x = float64(w - 1)
		}
		if y > float64(h-1) {
			y = float64(h - 1)
		}
		x0, y0 := int(x), int(y)
		fx, fy := x-float64(x0), y-float64(y0)
		var acc [3]float64
		for dy := 0; dy <= 1; dy++ {
			for dx := 0; dx <= 1; dx++ {
				px := min(x0+dx, w-1)
				py := min(y0+dy, h-1)
				pr, pg, pb, _ := img.At(b.Min.X+px, b.Min.Y+py).RGBA()
				wgt := (1 - math.Abs(float64(dx)-fx)) * (1 - math.Abs(float64(dy)-fy))
				acc[0] += float64(pr>>8) * wgt
				acc[1] += float64(pg>>8) * wgt
				acc[2] += float64(pb>>8) * wgt
			}
		}
		return acc[0], acc[1], acc[2]
	}
	// Inverse-map destination pixels through the similarity transform.
	inv := 1 / m.Scale
	// R is orthonormal: R^-1 = R^T.
	warp := func(px, py float64) (float64, float64) {
		qx := (px - m.T[0]) * inv
		qy := (py - m.T[1]) * inv
		return m.R[0][0]*qx + m.R[1][0]*qy, m.R[0][1]*qx + m.R[1][1]*qy
	}
	blob := make([]float32, 0, 3*RecInputSize*RecInputSize)
	for c := 0; c < 3; c++ {
		for y := 0; y < RecInputSize; y++ {
			for x := 0; x < RecInputSize; x++ {
				sx, sy := warp(float64(x), float64(y))
				r, g, bl := sample(sx, sy)
				var v float64
				switch c {
				case 0:
					v = bl // BGR order, matching the ArcFace reference
				case 1:
					v = g
				default:
					v = r
				}
				blob = append(blob, float32((v-127.5)/127.5))
			}
		}
	}
	return blob, nil
}
