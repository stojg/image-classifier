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
