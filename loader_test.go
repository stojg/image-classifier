package main

import (
	"bytes"
	"io"
	"testing"
)

func TestAsFloat64Slice(t *testing.T) {

	set := ImageSet{}
	preparedData := set.asTrainingSet()

	if len(preparedData) != 0 {
		t.Errorf("data should be empty from empty list")
	}

	image := CIFAR10Image{}
	io.Copy(&image, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image)

	preparedData = set.asTrainingSet()
	if len(preparedData) != 1 {
		t.Errorf("data should have 1 row of data, have %d", len(preparedData))
	}

	if len(preparedData[0]) != 2 {
		t.Errorf("Expected data[0] to have only have data and labels, got %d cols", len(preparedData[0]))
	}

	if len(preparedData[0][0]) != 8 {
		t.Errorf("Expected data[0][0] to have 8 data points, got %d ", len(preparedData[0][0]))
	}

	if len(preparedData[0][1]) != 1 {
		t.Errorf("Expected data[0][1] to have 1 class, got %d ", len(preparedData[0][1]))
	}

	image2 := CIFAR10Image{}
	io.Copy(&image2, bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}))
	set = append(set, image2)
	preparedData = set.asTrainingSet()

	if len(preparedData) != 2 {
		t.Errorf("data should have 2 rows of data, have %d", len(preparedData))
	}

}
