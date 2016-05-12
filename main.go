package main

import (
	"fmt"
	"log"
	"os"
)

func main() {

	x, y, err := loadDeployData()
	if err !=nil {
		panic(err)
	}

	fmt.Println(len(x), len(y))
	trainingX, trainingY := x[:15000], y[:15000]
	testX, testY := x[15000:], y[15000:]

//	// 50 000 images
//	trainingImages := loadCIFAR10("data/data_batch_*")
//	// 10 000 images
//	testImages := loadCIFAR10("data/test_batch.bin")
//	log.Printf("converting image data for classifier")
//
//	trainingX, trainingY := trainingImages.asFloatSlices()
//	testX, testY := testImages.asFloatSlices()

	if len(trainingX) < 1 || len(testY) < 1 {
		log.Printf("no training or test data found")
		os.Exit(1)
	}

	trainingLen := 15000
	testLen := 1500
	n := Normaliser{}
	trX := n.Normalise(trainingX[:trainingLen])
	trY := trainingY[:trainingLen]
	teX := n.Normalise(testX[:testLen])
	teY := testY[:testLen]

	log.Printf("training set size %d", len(trX))
	log.Printf("test set size %d", len(teX))

	nn := &NeuralNet{log: true}
	log.Printf("training neural net")
	nn.Train(trX, trY)

	log.Printf("predicting on neural net")

	var correct int
	for i := range teX {
		result := nn.Predict(teX[i])
		if teY[i][result[0]] > 0 {
			correct++
		}
	}
	log.Printf("neural net classifier accuracy: %0.1f%% (%d / %d)", percent(correct, len(teY)), correct, len(teY))
	fmt.Println(".")
}

func percent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}
