package main

import (
	"testing"
)

func TestTranspose(t *testing.T) {

	input := [][]float64{
		[]float64{1, 4, 7},
		[]float64{2, 5, 8},
		[]float64{3, 6, 9},
	}

	expected := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}

	norm := &Normaliser{}

	actual := norm.Transpose(input)

	if len(actual) != len(expected) {
		t.Errorf("wanted input to be the same length as output, wanted %d, got %d", len(expected), len(actual))
	}

	for i := range actual {
		if len(actual[i]) != len(expected[i]) {
			t.Errorf("wanted input to be the same length as output, wanted %d, got %d", len(expected[i]), len(actual[i]))
		}
		for j := range actual[i] {
			if actual[i][j] != expected[i][j] {
				t.Errorf("wanted %f, got %f", expected[i][j], actual[i][j])
			}
		}
	}
}

func TestNormalising(t *testing.T) {

	norm := &Normaliser{}

	input := [][]float64{
		[]float64{1, 4, 7, 345},
		[]float64{2, 5, 8, 345},
		[]float64{3, 6, 9, 214},
	}

	patterns := []byte{
		NormNumeric,
		NormNumeric,
		NormNoop,
		NormCategory,
	}

	expected := [][]float64{
		[]float64{-1, -1, 7, 1, -1},
		[]float64{0, 0, 8, 1, -1},
		[]float64{1, 1, 9, -1, 1},
	}

	actual := norm.Normalise(input, patterns)

	if len(actual) != len(expected) {
		t.Errorf("wanted actual to be the same length as output, wanted %d, got %d", len(expected), len(actual))
	}

	for i := range actual {
		if len(actual[i]) != len(expected[i]) {
			t.Errorf("wanted actual(%d) to be the same length (%d) as expected[%d], got %d\n", i, len(actual[i]), i, len(expected[i]))
		}
		for j := range actual[i] {
			if actual[i][j] != expected[i][j] {
				t.Errorf("Wanted expected[%d][%d] %f, got actual[%d][%d] %f", i, j, expected[i][j], i, j, actual[i][j])
			}
		}
	}
}
