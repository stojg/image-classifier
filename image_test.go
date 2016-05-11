package main

import (
	"bytes"
	"io"
	"testing"
)

func TestLoading(t *testing.T) {
	image := CIFAR10Image{}

	data := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9}
	io.Copy(&image, bytes.NewReader(data))

	if image.label != byte(data[0]) {
		t.Errorf("Expected label to be %d, got %d", data[0], image.label)
	}

	if image.raw[0] != float64(data[1]) {
		t.Errorf("Expected first pixel byte be %d, got %f", data[1], image.raw[1])
	}
	if image.raw[7] != float64(data[8]) {
		t.Errorf("Expected last pixel byte be %d, got %f", data[8], image.raw[7])
	}
	if image.raw[7] != float64(data[8]) {
		t.Errorf("Expected last pixel byte be %d, got %f", data[8], image.raw[7])
	}

}

func TestAsFloat64Slice(t *testing.T) {

	set := ImageSet{}
	x, y := set.asFloatSlices()

	if len(x) != 0 {
		t.Errorf("data should be empty from empty list")
	}

	image := CIFAR10Image{}
	io.Copy(&image, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image)

	x, y = set.asFloatSlices()
	if len(x) != 1 {
		t.Errorf("data should have 1 row of data, have %d", len(x))
	}

	if len(x[0]) != 8 {
		t.Errorf("Expected x to have 8 data points, got %d ", len(x[0]))
	}

	if len(y[0]) != 10 {
		t.Errorf("Expected y to have 10 classes, got %d ", len(y[0]))
	}

	image2 := CIFAR10Image{}
	io.Copy(&image2, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image2)
	x, y = set.asFloatSlices()

	if len(x) != 2 {
		t.Errorf("data should have 2 rows of data, have %d", len(x))
	}

}
