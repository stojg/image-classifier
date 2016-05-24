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
	trY := [][]byte{
		[]byte{1, 0},
		[]byte{1, 0},
		[]byte{0, 1},
		[]byte{0, 1},
	}

	neuro := NeuralNet{}

	// initialize parameters (weights) randomly for input -> hidden layer
	neuro.W1 = NewRandomMatrix(2, 4).ScalarMul(0.01)
	neuro.Bias1 = NewRandomMatrix(1, 4).ScalarMul(0.01)

	// initialize parameters (weights) randomly for hidden-> output layer
	neuro.W2 = NewRandomMatrix(4, 2).ScalarMul(0.01)
	neuro.Bias2 = NewRandomMatrix(1, 2).ScalarMul(0.01)

	xBatches, yBatches := neuro.getBatches(1, trX, trY)

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		catch, _, _, _ = neuro.GradientDescent(xBatches[0], yBatches[0])
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
	trY := [][]byte{
		[]byte{1, 0},
		[]byte{1, 0},
		[]byte{0, 1},
		[]byte{0, 1},
	}

	neuro := NeuralNet{}

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		neuro.Train(trX, trY, 10, 1, 50)
	}
	trailResult = catch
}
