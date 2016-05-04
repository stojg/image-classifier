package main

import (
	"math"
	"log"
)

func AbsSub(a, b []dValue) dValue {
	var result dValue
	for k := range a {
		diff := a[k] - b[k]
		if diff < 0 {
			diff = -diff
		}
		result += diff
	}
	return result
}

type NearestNeighbour struct {
	data   []dValue
	labels []dLabel
}

func (c *NearestNeighbour) Train(X []dValue, y []dLabel) {
	c.data = X
	c.labels = y
}

func (c *NearestNeighbour) Predict(testSet []dValue) []dLabel {
	const stride = 1024 * 3



	numTests := len(testSet) / stride
	numTraining := len(c.data) / stride

	betweenReport := numTests / 40
	lastReport := 0

	predictions := make([]dLabel, numTests)

	for i := 0; i < numTests; i++ {
		testData := testSet[i*stride : i*stride+stride]

		scores := make([]dValue, numTraining)
		for j := 0; j < numTraining; j++ {
			scores[j] = AbsSub(testData, c.data[j*stride:j*stride+stride])
		}

		lowestScore := dValue(math.MaxInt64)
		for s := 0; s < len(scores); s++ {
			if lowestScore > scores[s] {
				predictions[i] = c.labels[s]
				lowestScore = scores[s]
			}
		}
		// min_index = np.argmin(distances) # get the index with smallest distance
		// Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
		if(i >= lastReport + betweenReport) {
			lastReport = i
			log.Printf("%9.3f%% - %d out of %d done\n", float32(i)/float32(numTests), i, numTests)
		}

	}
	return predictions
}
