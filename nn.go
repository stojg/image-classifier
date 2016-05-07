package main

import (
	"math"
)

// NearestNeighbour predicts a by using the k nearest neighbour with k = 1
type NearestNeighbour struct {
	data [][][]float64
	log  bool
}

// Train trains the predictor
func (nn *NearestNeighbour) Train(x [][][]float64) {
	nn.data = x
}

// Predict nearest neighbour
func (nn *NearestNeighbour) Predict(input []float64) []float64 {

	scores := make([]float64, len(nn.data))
	for j := 0; j < len(nn.data); j++ {
		scores[j] = calculateDifference(input, nn.data[j][0])
	}

	prediction := make([]float64, len(nn.data[0][1]))
	// find closest neighbour
	lowestScore := float64(math.MaxInt64)
	for s := 0; s < len(scores); s++ {
		if lowestScore > scores[s] {
			copy(prediction, nn.data[s][1])
			lowestScore = scores[s]
		}
	}

	return prediction
}

func calculateDifference(a, b []float64) float64 {
	var result float64
	for k := range a {
		diff := a[k] - b[k]
		if diff < 0 {
			diff = -diff
		}
		result += diff
	}
	return result
}
