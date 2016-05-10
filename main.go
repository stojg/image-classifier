package main

import (
	"fmt"
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

	x := normalise(trainingData[:9000])
	y := normalise(trainingData[9000:10000])

	log.Printf("training set size %d", len(x))
	log.Printf("test set size %d", len(y))

	trailer(x, y)
	//nearest(x, y)
	fmt.Println(".")
}

func normalise(x [][][]float64) [][][]float64 {
	tempX := make([][]float64, len(x))
	for row := range x {
		tempX[row] = x[row][0]
	}

	n := &Normaliser{}
	normX := n.Normalise(tempX, []byte{0})
	for row := range x {
		x[row][0] = normX[row]
	}
	return x
}

func trailer(x, y [][][]float64) {

	nn := &trail{}
	log.Printf("training trail")
	nn.Train(x)

	log.Printf("predicting on trail")
	var correctPredictions int
	for _, p := range y {
		result := nn.Predict(p[0])
		expected := p[1]
		if equals(result, expected) {
			correctPredictions++
		}
	}
	log.Printf("Linear classifier accuracy: %0.1f%% (%d / %d)", percent(correctPredictions, len(y)), correctPredictions, len(y))
}

func nearest(x, y [][][]float64) {
	//n := &Normaliser{}
	//patterns := make([]byte, len(trainingData[0]))
	//n.Normalise(trainingData, patterns)

	//nn := &NearestNeighbour{}
	//log.Printf("training Nearest Neighbour")
	//nn.Train(x)
	//
	//log.Printf("predicting on Nearest Neighbour")
	//var correctPredictions int
	//for i, p := range y {
	//	result := nn.Predict(p[0])
	//	expected := p[1]
	//	if equals(result, expected) {
	//		correctPredictions++
	//	}
	//	if i%10 == 0 {
	//		log.Printf("accuracy: %.0f%% (%d / %d)", percent(correctPredictions, i+1), correctPredictions, i+1)
	//	}
	//}
	//log.Printf("Nearest Neighbour accuracy: %0.1f%% (%d / %d)", percent(correctPredictions, len(y)), correctPredictions, len(y))
}

func percent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}

func equals(actual []int, expected []float64) bool {
	var matches int
	for i := range actual {
		if int(expected[i]) == actual[i] {
			matches++
		}
	}
	if matches == len(expected) {
		return true
	}
	return false
}
