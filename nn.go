package main

import (
	"github.com/gonum/matrix/mat64"
	"log"
	"math"
)

type NearestNeighbour struct {
	data [][][]float64
	log  bool
}

func (nn *NearestNeighbour) Train(x [][][]float64) {
	nn.data = x
}

func (nn *NearestNeighbour) Predict(input [][]float64) *mat64.Dense {
	numTests := len(input)
	betweenReport := numTests / 40
	lastReport := 0

	predictions := mat64.NewDense(len(input), len(nn.data[0][1]), nil)
	for i := 0; i < numTests; i++ {
		scores := make([]float64, len(nn.data))
		for j := 0; j < len(nn.data); j++ {
			scores[j] = calculateDifference(input[i], nn.data[j][0])
		}

		// find closest neighbour
		lowestScore := float64(math.MaxInt64)
		for s := 0; s < len(scores); s++ {
			if lowestScore > scores[s] {
				predictions.Set(i, 0, nn.data[s][1][0])
				lowestScore = scores[s]
			}
		}

		// log progress
		if nn.log && i >= lastReport+betweenReport {
			lastReport = i
			log.Printf("%9.3f%% - %d out of %d done\n", float32(i)/float32(numTests), i, numTests)
		}
	}

	return predictions
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
