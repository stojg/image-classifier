package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"os"
	"time"
)

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	rawX, rawY, err := wineLoader("testdata/wine.data")
	if err != nil {
		panic(err)
	}

	// ensure that the data is randomised
	for i := range rawX {
		j := rand.Intn(i + 1)
		rawX[i], rawX[j] = rawX[j], rawX[i]
		rawY[i], rawY[j] = rawY[j], rawY[i]
	}

	// normalise the data into a standard deviation (roughly between -1 to +1) with a gaussian distribution
	log.Printf("normalising data")
	n := Normaliser{}
	normX := n.StdDev(rawX)

	// 80% goes to the training, 20% to test
	trainingSize := int(float64(len(rawX)) * 0.8)

	trX := normX[:trainingSize]
	trY := rawY[:trainingSize]
	log.Printf("training set contains %d examples of dimensions X: %d and Y: %d", len(trX), len(trX[0]), len(trY[0]))

	teX := normX[trainingSize:]
	teY := rawY[trainingSize:]
	log.Printf("test set contains %d examples of dimensions X: %d and Y: %d", len(teX), len(teX[0]), len(teY[0]))

	nn := &NeuralNet{
		HiddenNeurons: 2000,
		Alpha:         1e-3,
		Lambda:        1e-3,
		numBatches:    8,
		numEpochs:     1000,
		log:           true,
		plot:          true,
	}

	// divide the training data into 50% training and 50% validation
	trX, trY, cvX, cvY := nn.Divide(trX, trY)

	log.Printf("training neural net")
	trainingError, cvError := nn.Train(trX, trY, cvX, cvY)
	log.Printf("final cost:\t%f\t%f", trainingError, cvError)

	// use the learned the net to predict and print the accuracy
	correct, acc := predict(nn, trX, trY)
	log.Printf("training accuracy: %0.1f%% (%d / %d)", acc, correct, len(trY))
	correct, acc = predict(nn, cvX, cvY)
	log.Printf("validation accuracy: %0.1f%% (%d / %d)", acc, correct, len(cvY))
	correct, acc = predict(nn, teX, teY)
	log.Printf("test accuracy: %0.1f%% (%d / %d)", acc, correct, len(teY))
	Save("learned_net.json", nn)
}

func predict(nn *NeuralNet, X, Y [][]float64) (correct int, percent float64) {
	for i := range X {
		result := nn.Predict(X[i])
		if Y[i][result[0]] > 0 {
			correct++
		}
	}
	if correct != 0 {
		percent = (float64(correct) / float64(len(X))) * 100
	}
	return correct, percent
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
