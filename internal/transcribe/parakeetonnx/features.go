package parakeetonnx

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

const (
	NFFT              = 512
	WinLength         = 400
	HopLength         = 160
	Preemphasis       = 0.97
	LogZeroGuardValue = float32(1.0 / (1 << 24))
	FeatureBins       = 128
)

type Features struct {
	Values [][]float32
	Length int64
}

func BuildFeatures(samples []float32) (Features, error) {
	if len(samples) == 0 {
		return Features{}, fmt.Errorf("audio samples are empty")
	}
	processed := preemphasize(samples)
	padded := zeroPadCenter(processed, NFFT/2)
	window := paddedHannWindow()
	melBank := melScaleFBanks(NFFT/2+1, 0, SampleRate/2, FeatureBins, SampleRate)
	fft := fourier.NewFFT(NFFT)

	frameCount := 1 + (len(padded)-NFFT)/HopLength
	if frameCount <= 0 {
		frameCount = 1
	}
	out := make([][]float32, FeatureBins)
	for i := range out {
		out[i] = make([]float32, frameCount)
	}

	frame := make([]float64, NFFT)
	featuresLength := int64(len(samples) / HopLength)
	if featuresLength < 1 {
		featuresLength = 1
	}

	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * HopLength
		for i := 0; i < NFFT; i++ {
			frame[i] = float64(padded[start+i] * window[i])
		}
		coeffs := fft.Coefficients(nil, frame)
		power := make([]float64, NFFT/2+1)
		for i := range power {
			realPart := real(coeffs[i])
			imagPart := imag(coeffs[i])
			power[i] = realPart*realPart + imagPart*imagPart
		}
		for mel := 0; mel < FeatureBins; mel++ {
			sum := 0.0
			for bin := 0; bin < len(power); bin++ {
				sum += power[bin] * melBank[bin][mel]
			}
			out[mel][frameIndex] = float32(math.Log(sum + float64(LogZeroGuardValue)))
		}
	}

	normalize(out, int(featuresLength))
	return Features{Values: out, Length: featuresLength}, nil
}

func preemphasize(samples []float32) []float32 {
	out := make([]float32, len(samples))
	if len(samples) == 0 {
		return out
	}
	out[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - Preemphasis*samples[i-1]
	}
	return out
}

func zeroPadCenter(samples []float32, pad int) []float32 {
	out := make([]float32, len(samples)+pad*2)
	copy(out[pad:], samples)
	return out
}

func paddedHannWindow() []float32 {
	window := make([]float32, NFFT)
	padLeft := NFFT/2 - WinLength/2
	for i := 0; i < WinLength; i++ {
		window[padLeft+i] = float32(0.5 - 0.5*math.Cos((2*math.Pi*float64(i))/float64(WinLength-1)))
	}
	return window
}

func normalize(features [][]float32, validFrames int) {
	if validFrames <= 1 {
		for mel := range features {
			for t := range features[mel] {
				if t >= validFrames {
					features[mel][t] = 0
				}
			}
		}
		return
	}
	for mel := range features {
		mean := float64(0)
		for t := 0; t < validFrames && t < len(features[mel]); t++ {
			mean += float64(features[mel][t])
		}
		mean /= float64(validFrames)

		variance := 0.0
		for t := 0; t < validFrames && t < len(features[mel]); t++ {
			delta := float64(features[mel][t]) - mean
			variance += delta * delta
		}
		variance /= float64(validFrames - 1)
		stddev := math.Sqrt(variance) + 1e-5

		for t := range features[mel] {
			if t < validFrames {
				features[mel][t] = float32((float64(features[mel][t]) - mean) / stddev)
			} else {
				features[mel][t] = 0
			}
		}
	}
}

func hzToMel(freq float64) float64 {
	if freq < 1000 {
		return 3 * freq / 200.0
	}
	return 15 + 27*math.Log(freq/1000.0+math.SmallestNonzeroFloat32)/math.Log(6.4)
}

func melToHz(mel float64) float64 {
	if mel < 15 {
		return 200 * mel / 3.0
	}
	return 1000 * math.Pow(6.4, (mel-15)/27.0)
}

func melScaleFBanks(nFreqs int, fMin float64, fMax float64, nMels int, sampleRate int) [][]float64 {
	if fMax <= 0 {
		fMax = float64(sampleRate) / 2
	}
	allFreqs := make([]float64, nFreqs)
	for i := range allFreqs {
		allFreqs[i] = float64(i) * float64(sampleRate/2) / float64(nFreqs-1)
	}
	mMin := hzToMel(fMin)
	mMax := hzToMel(fMax)
	mPts := make([]float64, nMels+2)
	for i := range mPts {
		mPts[i] = melToHz(mMin + (mMax-mMin)*float64(i)/float64(nMels+1))
	}

	fb := make([][]float64, nFreqs)
	for i := 0; i < nFreqs; i++ {
		fb[i] = make([]float64, nMels)
	}
	for mel := 0; mel < nMels; mel++ {
		denomUp := mPts[mel+1] - mPts[mel]
		denomDown := mPts[mel+2] - mPts[mel+1]
		for i, freq := range allFreqs {
			up := (freq - mPts[mel]) / denomUp
			down := (mPts[mel+2] - freq) / denomDown
			value := math.Max(0, math.Min(up, down))
			value *= 2.0 / (mPts[mel+2] - mPts[mel])
			fb[i][mel] = value
		}
	}
	return fb
}
