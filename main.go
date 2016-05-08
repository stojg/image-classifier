package main

import (
	"log"
	"os"
)

func main() {
	trainingImgs, _ := loadCIFAR10("data/data_batch_*")
	testImgs, _ := loadCIFAR10("data/test_batch.bin")
	log.Printf("converting image data for classifier")

	trainingData := trainingImgs.asMatrix()
	testData := testImgs.asMatrix()

	if len(trainingData) < 1 || len(testData) < 1 {
		log.Printf("no training or test data found")
		os.Exit(1)
	}
	log.Printf("training set full size %d", len(trainingData))

	x := trainingData[:40000]
	y := trainingData[40000:]

	//n := &Normaliser{}
	//patterns := make([]byte, len(trainingData[0]))
	//n.Normalise(trainingData, patterns)

	log.Printf("training set size %d", len(x))
	log.Printf("test set size %d", len(y))

	nn := &NearestNeighbour{}
	log.Printf("training Nearest Neighbour")
	nn.Train(x)

	log.Printf("predicting on Nearest Neighbour")
	var correctPredictions int
	for i, p := range y {
		result := nn.Predict(p[0])
		expected := p[1]
		if equals(result, expected) {
			correctPredictions++
		}
		if i%10 == 0 {
			log.Printf("accuracy: %.0f%% (%d / %d) out of %d tests", percent(correctPredictions, i+1), correctPredictions, i+1, len(y))
		}
	}
	log.Printf("Nearest Neighbour accuracy: %0.1f%% (%d / %d)\n", percent(correctPredictions, len(testData)), correctPredictions, len(testData))

}

func percent(a, b int) float64 {
	if b==0 {
		return 0
	}
	return float64(a)/float64(b)*100
}

func equals(actual, expected []float64) bool {
	var matches int
	for i := range actual {
		if expected[i] == actual[i] {
			matches++
		}
	}
	if matches == len(expected) {
		return true
	}
	return false
}
