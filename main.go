package main

import (
	"log"
	//	"math"
	"os"
)

func main() {
	trainingImgs, _ := loadCIFAR10("data/data_batch_*")
	testImgs, _ := loadCIFAR10("data/test_batch.bin")
	log.Printf("Converting image data for classifier")
	trainingData := trainingImgs.asMatrix()
	testData := testImgs.asMatrix()
	if len(trainingData) < 1 || len(testData) < 1 {
		log.Printf("No training or test data found")
		os.Exit(1)
	}

	trainingData = trainingData[:1000]
	testData = trainingData[1000:1100]

	log.Printf("training set size %d", len(trainingData))
	log.Printf("test set size %d", len(testData))

	nn := &NearestNeighbour{log: true}
	log.Printf("Training Nearest neighbour")
	nn.Train(trainingData)
	log.Printf("Predicting on Nearest neighbour")

	var correctPredictions float64
	for _, p := range testData {
		result := nn.Predict(p[0])
		expected := p[1]
		matches := 0
		for i := range result {
			if expected[i] == result[i] {
				matches++
			}
		}
		if matches == len(expected) {
			correctPredictions++
		}
	}

	log.Printf("Nearest Neighbour accuracy: %0.1f%% (%.0f/%d)\n", correctPredictions/float64(len(testData))*100, correctPredictions, len(testData))

}
