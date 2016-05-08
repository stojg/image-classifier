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

	if image.label != float64(data[0]) {
		t.Errorf("Expected label to be %d, got %.0f", data[0], image.label)
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
	preparedData := set.asMatrix()

	if len(preparedData) != 0 {
		t.Errorf("data should be empty from empty list")
	}

	image := CIFAR10Image{}
	io.Copy(&image, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image)

	preparedData = set.asMatrix()
	if len(preparedData) != 1 {
		t.Errorf("data should have 1 row of data, have %d", len(preparedData))
	}

	if len(preparedData[0]) != 2 {
		t.Errorf("Expected data[0] to have only have data and labels, got %d cols", len(preparedData[0]))
	}

	if len(preparedData[0][0]) != 8 {
		t.Errorf("Expected data[0][0] to have 8 data points, got %d ", len(preparedData[0][0]))
	}

	if len(preparedData[0][1]) != 10 {
		t.Errorf("Expected data[0][1] to have 10 classes, got %d ", len(preparedData[0][1]))
	}

	image2 := CIFAR10Image{}
	io.Copy(&image2, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image2)
	preparedData = set.asMatrix()

	if len(preparedData) != 2 {
		t.Errorf("data should have 2 rows of data, have %d", len(preparedData))
	}

}
