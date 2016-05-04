package main

import (
	"log"
)

type dValue int64
type dLabel byte

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
	log.Printf("Converting image data for classifier")
	xTraining := trainingSet.RawData()
	xTest := testSet.RawData()

	nn := &NearestNeighbour{}
	log.Printf("Training classifier")
	nn.Train(xTraining, trainingLabels)
	log.Printf("Predicting")
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
