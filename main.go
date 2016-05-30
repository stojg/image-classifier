package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"os"
	"time"
)

func main() {

	//trainingX, trainingY, err := cifar10Loader("data/data_batch_1*")
	trainingX, trainingY, err := wineLoader("testdata/wine.data")
	if err != nil {
		panic(err)
	}

	rand.Seed(time.Now().UTC().UnixNano())
	//rand.Seed(0)

	for i := range trainingX {
		j := rand.Intn(i + 1)
		trainingX[i], trainingX[j] = trainingX[j], trainingX[i]
		trainingY[i], trainingY[j] = trainingY[j], trainingY[i]
	}

	// 80% goes to the training
	trainingSize := int(float64(len(trainingX)) * 0.8)
	log.Printf("normalising data")
	n := Normaliser{}
	// this normalises the data into a standard deviation, roughly between -1 to +1 with a guassian distribution
	trX := n.StdDev(trainingX[:trainingSize])
	trY := trainingY[:trainingSize]
	log.Printf("training set contains %d examples of dimensions X: %d and Y: %d", len(trX), len(trX[0]), len(trY[0]))

	teX := n.StdDev(trainingX[trainingSize:])
	teY := trainingY[trainingSize:]
	log.Printf("test set contains %d examples of dimensions X: %d and Y: %d", len(teX), len(teX[0]), len(teY[0]))

	nn := &NeuralNet{
		log:    true,
		plot:   true,
		lambda: 0.01,
		alpha:  0.001,
	}

	log.Printf("training neural net")
	// train the network with n epochs
	trainingError, cvError := nn.Train(trX, trY, 40, 5000, 8)

	log.Printf("errors:")
	log.Printf("\ttraining\t%0.5f", trainingError)
	log.Printf("\tvalidation\t%0.5f", cvError)

	var correct int
	for i := range trX {
		result := nn.Predict(trX[i])
		if trY[i][result[0]] > 0 {
			correct++
		}
	}
	log.Printf("training accuracy: %0.1f%% (%d / %d)", percent(correct, len(trY)), correct, len(trY))

	correct = 0
	for i := range teX {
		result := nn.Predict(teX[i])
		if teY[i][result[0]] > 0 {
			correct++
		}
	}
	log.Printf("test accuracy: %0.1f%% (%d / %d)", percent(correct, len(teY)), correct, len(teY))

	Save("wine.dat", nn)
}

func percent(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b) * 100
}

func Save(fileName string, t *NeuralNet) {
	out_f, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("failed to dump the network to " + fileName)
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	err = encoder.Encode(t)
	if err != nil {
		panic(err)
	}
}

func Load(fileName string) *NeuralNet {
	in_f, err := os.Open(fileName)
	if err != nil {
		panic("failed to load " + fileName)
	}
	defer in_f.Close()
	decoder := json.NewDecoder(in_f)
	nn := &NeuralNet{}
	err = decoder.Decode(nn)
	if err != nil {
		panic(err)
	}
	return nn
}
