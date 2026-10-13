package retrievalbench

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestParseFlickrCSV(t *testing.T) {
	input := `sentids,imgid,split,filename,tokens,raw,sentid
"[1, 2]","[0, 0]",test,a.jpg,"[]","[""A dog runs."", ""A dog, outside.""]","[1, 2]"
"[3]","[1]",test,b.jpg,"[]","[""A cat sleeps.""]","[3]"
`
	items, err := ParseFlickrCSV(strings.NewReader(input))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := []Item{
		{Image: "a.jpg", Captions: []string{"A dog runs.", "A dog, outside."}},
		{Image: "b.jpg", Captions: []string{"A cat sleeps."}},
	}
	if !reflect.DeepEqual(items, want) {
		t.Fatalf("items = %#v, want %#v", items, want)
	}
}

func TestParseFlickrCSVRejectsMissingColumns(t *testing.T) {
	if _, err := ParseFlickrCSV(strings.NewReader("filename\na.jpg\n")); err == nil {
		t.Fatal("expected error for missing raw column")
	}
}

func TestParseFlickrCSVRejectsItemWithoutCaptions(t *testing.T) {
	input := "filename,raw\na.jpg,\"[]\"\n"
	if _, err := ParseFlickrCSV(strings.NewReader(input)); err == nil {
		t.Fatal("expected error for image without captions")
	}
}

func TestEvaluateRecall(t *testing.T) {
	images := [][]float32{{1, 0}, {0, 1}}
	captions := [][]float32{
		{0.9, 0.1}, // image 0, rank 0
		{0.2, 0.8}, // image 0, rank 1
		{0.1, 0.9}, // image 1, rank 0
	}
	captionImage := []int{0, 0, 1}

	got := Evaluate(images, captions, captionImage, []int{1, 2})

	wantT2I := map[int]float64{1: 2.0 / 3.0, 2: 1}
	wantI2T := map[int]float64{1: 1, 2: 1}
	assertRecall(t, "text-to-image", got.TextToImage, wantT2I)
	assertRecall(t, "image-to-text", got.ImageToText, wantI2T)
}

func TestEvaluateImageToTextUsesBestCaptionRank(t *testing.T) {
	images := [][]float32{{1, 0}, {0, 1}}
	captions := [][]float32{
		{0, 1},     // image 0, scores highest for image 1
		{0.6, 0.4}, // image 0
		{0.5, 0.5}, // image 1
	}
	captionImage := []int{0, 0, 1}

	got := Evaluate(images, captions, captionImage, []int{1})

	// Image 0 ranks caption 1 first; image 1 ranks caption 0 (not its own) first.
	assertRecall(t, "image-to-text", got.ImageToText, map[int]float64{1: 0.5})
}

func TestEvaluateCountsTiesAgainstTarget(t *testing.T) {
	images := [][]float32{{1, 0}, {1, 0}}
	captions := [][]float32{{1, 0}}
	got := Evaluate(images, captions, []int{1}, []int{1})
	if got.TextToImage[1] != 0 {
		t.Fatalf("tied distractor must not count as a hit, got %v", got.TextToImage[1])
	}
}

func TestTruncateNormalize(t *testing.T) {
	got := TruncateNormalize([]float32{3, 4, 12}, 2)
	want := []float32{0.6, 0.8}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestTruncateNormalizeKeepsFullLength(t *testing.T) {
	got := TruncateNormalize([]float32{3, 4}, 0)
	if len(got) != 2 {
		t.Fatalf("expected full length, got %v", got)
	}
}

func assertRecall(t *testing.T, name string, got map[int]float64, want map[int]float64) {
	t.Helper()
	for k, w := range want {
		if math.Abs(got[k]-w) > 1e-9 {
			t.Fatalf("%s R@%d = %v, want %v", name, k, got[k], w)
		}
	}
}
