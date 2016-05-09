package main

import (
	"math"
)

// NearestNeighbour predicts a by using the k nearest neighbour with k = 1
type NearestNeighbour struct {
	labels   [][]float64
	training []*Matrix
}

// Train trains the predictor
func (nn *NearestNeighbour) Train(x [][][]float64) {
	nn.training = make([]*Matrix, len(x))
	nn.labels =make([][]float64, len(x))
	for i := range x {
		nn.training[i] = NewMatrixF(x[i][0], 1, len(x[i][0]))
		nn.labels[i] = x[i][1]
	}
}

// Predict nearest neighbour
func (nn *NearestNeighbour) Predict(input []float64) []float64 {

	testData := NewMatrixF(input, 1, len(input))

	scores := make([]float64, len(nn.labels))


	for i, val := range nn.training {
		scores[i] = val.Sub(testData).AbsSum()
	}

	k := 1

	var neighbours [][]float64
	// find closest neighbour
	lowestScore := math.Inf(1)
	var lowestIndex int
	for n := 0; n < k; n++ {
		for s := 0; s < len(scores); s++ {
			if lowestScore > scores[s] {
				lowestIndex = s
				lowestScore = scores[s]
			}
		}

		neighbours = append(neighbours, nn.labels[lowestIndex])
		// we don't have enough test images to fill out all neighbours
		// @todo return error and the list so far
		if len(scores) < 2 {
			break
		}
		// delete this entry from the scores
		scores[n] = scores[len(scores)-1]
		scores = scores[:len(scores)-1]
	}

	// @todo: let all neighbours vote on which class input is closest to
	// @todo: idea, make neighbours a struct to hold score and labels
	return neighbours[0]
}

// L1 is the Manhattan cost (error) function
func (nn *NearestNeighbour) L1(a, b []float64) float64 {
	var result float64
	var diff float64
	for k := range a {
		diff = a[k] - b[k]
		// faster math.Abs
		if diff < 0 {
			diff = -diff
		}
		result += diff // sum the difference
	}
	return result
}

// L2 is the Euclidean cost (error) function
func (nn *NearestNeighbour) L2(a, b []float64) float64 {
	var result float64
	var diff float64
	for k := range a {
		diff = (a[k] - b[k]) * (a[k] - b[k])
		// faster math.Abs
		if diff < 0 {
			diff = -diff
		}
		result += diff // sum the difference
	}
	return (result)
}
