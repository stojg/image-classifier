package main

import "testing"

var trailResult *Matrix

func BenchmarkTrailTrain(b *testing.B) {

	trX := [][]float64{
		[]float64{0, 1},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 0},
	}
	trY := [][]byte{
		[]byte{1, 0},
		[]byte{1, 0},
		[]byte{0, 1},
		[]byte{0, 1},
	}

	trail := NeuralNet{}

	for i := 0; i < b.N; i++ {
		trail.Train(trX, trY, 10)
	}
	trailResult = trail.hiddenBias
}
