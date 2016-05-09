package main

import (
	"math"
	"testing"
)

var linResult float64

func TestSVMLoss(t *testing.T) {
	test := NewVector([]float64{-15, 22, -44, 56, 1}) // last entry is the bias "compensation"
	weights := NewMatrix([][]float64{
		[]float64{0.01, -0.05, 0.1, 0.05, 0}, // last entry is the bias values
		[]float64{0.7, 0.2, 0.05, 0.16, 0.2},
		[]float64{0.0, -0.45, -0.2, 0.03, -0.3},
	})

	actual := SVMLoss(test, 2, weights)
	expected := float64(1.580001)

	diff := math.Abs(actual - expected)
	if diff > 0.0001 {
		t.Errorf("expected %f, got %f, difference %f", expected, actual, diff)
	}
}

func BenchmarkSVMLoss(b *testing.B) {
	test := NewVector([]float64{1, 2, 3, 1}) // last entry is the bias "compensation"

	weights := NewMatrix([][]float64{
		[]float64{1, 2, 3, 1}, // last entry is the bias
		[]float64{4, 5, 6, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
	})

	var actual float64
	for i := 0; i < b.N; i++ {
		actual = SVMLoss(test, 1, weights)
	}
	linResult = actual
}

func TestSoftMaxLossNotCorrect(t *testing.T) {
	test := NewVector([]float64{-15, 22, -44, 56, 1}) // last entry is the bias "compensation"
	weights := NewMatrix([][]float64{
		[]float64{0.01, -0.05, 0.1, 0.05, 0}, // last entry is the bias values
		[]float64{0.7, 0.2, 0.05, 0.16, 0.2},
		[]float64{0.0, -0.45, -0.2, 0.03, -0.3},
	})

	actual := SoftMaxLoss(test, 0, weights)
	expected := float64(4.170191)

	diff := math.Abs(actual - expected)
	if diff > 0.01 {
		t.Errorf("expected %f, got %f, difference %.12f", expected, actual, diff)
	}

	// this is the most likely candidate
	actual = SoftMaxLoss(test, 1, weights)
	expected = float64(0.460191)

	diff = math.Abs(actual - expected)
	if diff > 0.01 {
		t.Errorf("expected %f, got %f, difference %.12f", expected, actual, diff)
	}

	actual = SoftMaxLoss(test, 2, weights)
	expected = float64(1.04)

	diff = math.Abs(actual - expected)
	if diff > 0.01 {
		t.Errorf("expected %f, got %f, difference %.12f", expected, actual, diff)
	}
}

func BenchmarkSoftMaxLoss(b *testing.B) {
	test := NewVector([]float64{1, 2, 3, 1}) // last entry is the bias "compensation"

	weights := NewMatrix([][]float64{
		[]float64{1, 2, 3, 1}, // last entry is the bias
		[]float64{4, 5, 6, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
		[]float64{7, 8, 9, 1},
	})

	var actual float64
	for i := 0; i < b.N; i++ {
		actual = SoftMaxLoss(test, 2, weights)
	}
	linResult = actual
}
