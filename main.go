package main

import (
	"fmt"
	"log"
	"math"
)

type dValue int64
type dLabel byte

type NearestNeighbour struct {
	data   []dValue
	labels []dLabel
}

func (c *NearestNeighbour) Train(X []dValue, y []dLabel) {
	c.data = X
	c.labels = y
}

func (c *NearestNeighbour) Predict(testSet []dValue) []dLabel {

	stride := 1024 * 3
	numTests := len(testSet) / stride
	numTraining := len(c.data) / stride

	prediction := make([]dLabel, numTests)
	for i := 0; i < numTests; i++ {
		// distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
		testData := testSet[i*stride : i*stride+stride]
		lowestScore := math.MaxInt32
		for j := 0; j < numTraining; j++ {
			result := AbsSub(testData, c.data[j*stride:j*stride+stride])
			if result < lowestScore {
				lowestScore = result
				prediction[i] = c.labels[j]
			}
		}
		// min_index = np.argmin(distances) # get the index with smallest distance
		// Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
		fmt.Printf(".")
	}
	return prediction
}

func AbsSub(a, b []dValue) int {
	var result dValue
	for k := range a {
		diff := dValue(a[k]) - dValue(b[k])
		if diff < 0 {
			diff = -diff
			//fmt.Printf("low %d - %d res: %d\n", a[k], b[k], diff)
		}
		result += (diff)
	}
	return int(result)
}

// compares two slices with the compare func
func mean(xS, yS []dLabel, compareFunc func(x, y dLabel) bool) float32 {
	count := 0
	for i := range xS {
		if compareFunc(xS[i], yS[i]) {
			count++
		}
	}
	return float32(count) / float32(len(xS))
}

func main() {
	trainingSet, trainingLabels := loadCIFAR10("data/data_batch_*")
	testSet, testLabels := loadCIFAR10("data/test_batch.bin")

	xTraining := trainingSet.RawData()
	xTest := testSet.RawData()

	nn := &NearestNeighbour{}
	nn.Train(xTraining, trainingLabels)
	Yte_predict := nn.Predict(xTest)

	avg := mean(Yte_predict, testLabels, func(x, y dLabel) bool {
		return x == y
	})
	log.Printf("accuracy: %f", avg)
}

type Trainer interface {
	Train(trainingImages []dValue, labels []dValue)
}

type Predictor interface {
	Predict(testData []dValue) (testLabels []dValue)
}
