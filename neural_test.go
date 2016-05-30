package main

import "testing"

var trailResult *Matrix

func BenchmarkTrailGradientDescent(b *testing.B) {

	trX := [][]float64{
		[]float64{0, 1},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 0},
	}
	trY := [][]float64{
		[]float64{1, 0},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 1},
	}

	neuro := NeuralNet{}

	hidden := 5
	// initialize parameters (weights) randomly for input -> hidden layer
	neuro.W1 = NewRandomMatrix(hidden, 2+1).ScalarMul(0.12)

	// initialize parameters (weights) randomly for hidden-> output layer
	neuro.W2 = NewRandomMatrix(2, hidden+1).ScalarMul(0.12)

	xBatches, yBatches := neuro.randomisedBatches(1, trX, trY)

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		_, catch, _ = neuro.miniBatch(xBatches[0], yBatches[0])
	}
	trailResult = catch
}

func BenchmarkTrailTrain(b *testing.B) {

	trX := [][]float64{
		[]float64{0, 1},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 0},
	}
	trY := [][]float64{
		[]float64{1, 0},
		[]float64{1, 0},
		[]float64{0, 1},
		[]float64{0, 1},
	}

	neuro := NeuralNet{}

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		neuro.Train(trX, trY, 10, 10, 1)
	}
	trailResult = catch
}
