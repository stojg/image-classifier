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

	neuro := &NeuralNet{
		HiddenNeurons: 2000,
		Alpha:         1e-1,
		Lambda:        1e-1,
		numBatches:    1,
		numEpochs:     10,
		log:           false,
		plot:          false,
	}

	// initialize parameters (weights) randomly for input -> hidden layer
	neuro.W1 = NewRandomMatrix(neuro.HiddenNeurons, 2+1).ScalarMul(0.12)

	// initialize parameters (weights) randomly for hidden-> output layer
	neuro.W2 = NewRandomMatrix(2, neuro.HiddenNeurons+1).ScalarMul(0.12)

	xBatches, yBatches := neuro.randomisedBatches(1, trX, trY)

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		_, catch, _ = neuro.costFunction(xBatches[0], yBatches[0], 0)
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

	neuro := &NeuralNet{
		HiddenNeurons: 2000,
		Alpha:         1e-1,
		Lambda:        1e-1,
		numBatches:    1,
		numEpochs:     10,
		log:           false,
		plot:          false,
	}

	var catch *Matrix
	for i := 0; i < b.N; i++ {
		neuro.Train(trX, trY, trX, trY)
	}
	trailResult = catch
}
