package main

import (
	"log"
	"os"
)

func main() {
	cTraining, _ := loadCIFAR10("data/data_batch_*")
	cTest, cTestLabels := loadCIFAR10("data/test_batch.bin")

	if len(cTraining) < 1 || len(cTest) < 1 {
		log.Printf("No training or test data found")
		os.Exit(1)
	}

	log.Printf("Converting image data for classifier")
	trainingData := cTraining.asTrainingSet()
	testData := cTest.asTestSet()

	nn := &NearestNeighbour{log: true}

	log.Printf("Training classifier")
	nn.Train(trainingData)

	log.Printf("Predicting")
	predictions := nn.Predict(testData)

	avg := labelCompare(predictions, cTestLabels, func(x, y float64) bool {
		return x == y
	})
	log.Printf("accuracy: %f", avg)
}
