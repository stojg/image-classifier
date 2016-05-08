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
	log.Printf("training set full size %d", len(trainingData))

	x := trainingData[:40000]
	y := trainingData[40000:]

	//n := &Normaliser{}
	//patterns := make([]byte, len(trainingData[0]))
	//n.Normalise(trainingData, patterns)

	log.Printf("training set size %d", len(x))
	log.Printf("veridication set size %d", len(y))

	nn := &NearestNeighbour{log: true}
	log.Printf("Training Nearest neighbour")
	nn.Train(x)
	log.Printf("Predicting on Nearest neighbour")

	every := 10

	var correctPredictions float64
	for i, p := range y {
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
		if i%every == 0 {
			log.Printf("%d/%d done, acc: %.0f%% %.0f/%d", i, len(y), correctPredictions/float64(i)*100, correctPredictions, i)
		}

	}

	log.Printf("Nearest Neighbour accuracy: %0.1f%% (%.0f/%d)\n", correctPredictions/float64(len(testData))*100, correctPredictions, len(testData))

}
