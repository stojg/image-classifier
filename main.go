package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	trainingImgs, _ := loadCIFAR10("data/data_batch_1*")
	testImgs, _ := loadCIFAR10("data/test_batch.bin")
	log.Printf("converting image data for classifier")

	trainingX, trainingY := trainingImgs.asFloatSlices()
	testX, testY := testImgs.asFloatSlices()

	if len(trainingX) < 1 || len(testY) < 1 {
		log.Printf("no training or test data found")
		os.Exit(1)
	}
	n := Normaliser{}
	trX := n.Normalise(trainingX[:1000])
	trY := trainingY[:1000]
	teX := n.Normalise(testX[9000:10000])
	teY := testY[9000:10000]

	log.Printf("training set size %d", len(trX))
	log.Printf("test set size %d", len(teX))

	neuralNet(trX, trY, teX, teY)
	fmt.Println(".")
}

func neuralNet(trX [][]float64, trY [][]byte, teX [][]float64, teY [][]byte) {
	nn := &trail{
		log: true,
	}
	log.Printf("training trail")
	nn.Train(trX, trY, 1000)

	log.Printf("predicting on trail")
	var correct int
	for i, p := range teX {
		result := nn.Predict(p)
		if teY[i][result[0]] > 0 {
			correct++
		}
	}
	log.Printf("neural net classifier accuracy: %0.1f%% (%d / %d)", percent(correct, len(teY)), correct, len(teY))
}

func percent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}
