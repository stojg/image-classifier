package main

import (
	"log"
)

func main() {

	trainingX, trainingY, err := wineLoader("testdata/wine.data")
	if err != nil {
		panic(err)
	}

	log.Printf("normalising data")
	n := Normaliser{}
	// this normalises the data into a standard deviation, roughly between -1 to +1 with a guassian distribution
	trX := n.Normalise(trainingX)
	trY := trainingY
	//teX := n.Normalise(testX[:testLen])
	//teY := testY[:testLen]

	log.Printf("training set size %d", len(trX))
	log.Printf("training set dimensions X: %d and Y: %d", len(trX[0]), len(trY[0]))

	nn := &NeuralNet{log: true, plot: true}

	log.Printf("training neural net")
	// train the network with n epochs
	nn.Train(trX, trY, 10000)

	//log.Printf("test set size %d", len(teX))
	//log.Printf("predicting on neural net")
	//
	//var correct int
	//for i := range teX {
	//	result := nn.Predict(teX[i])
	//	if teY[i][result[0]] > 0 {
	//		correct++
	//	}
	//}
	//log.Printf("neural net classifier accuracy: %0.1f%% (%d / %d)", percent(correct, len(teY)), correct, len(teY))
}

func junk() {
	// the image data is downloaded from https://www.cs.toronto.edu/~kriz/cifar.html (binary version) and chucked into
	// a "data" folder.
	// 50 000 images
	//trainingImages := loadCIFAR10("data/data_batch_1*")
	// 10 000 images
	//testImages := loadCIFAR10("data/test_batch.bin")
	//log.Printf("converting image data for classifier")

	// the "images" as converted back into a []float64 for both X (pixeldata) and Y(labels)
	//trainingX, trainingY := trainingImages.asFloatSlices()
	//testX, testY := testImages.asFloatSlices()
	//
	//if len(trainingX) < 1 || len(testY) < 1 {
	//	log.Printf("no training or test data found")
	//	os.Exit(1)
	//}
	//
	//trainingLen := 1000
	//testLen := 100
}

func percent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}
