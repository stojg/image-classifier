package main

import (
	"encoding/json"
	"log"
	"math/rand"
	"os"
	"time"
)

func main() {

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

	log.Printf("normalising data")
	n := Normaliser{}
	// this normalises the data into a standard deviation, roughly between -1 to +1 with a guassian distribution
	trX := n.StdDev(trainingX[:150])
	trY := trainingY[:150]
	log.Printf("training set size %d", len(trX))
	log.Printf("training set dimensions X: %d and Y: %d", len(trX[0]), len(trY[0]))

	teX := n.StdDev(trainingX[150:])
	teY := trainingY[150:]
	log.Printf("test set size %d", len(teX))
	log.Printf("test set dimensions X: %d and Y: %d", len(teX[0]), len(teY[0]))

	nn := &NeuralNet{log: true, plot: true}

	log.Printf("training neural net")
	// train the network with n epochs
	nn.Train(trX, trY, 5000, 8, 10)

	log.Printf("test set size %d", len(teX))
	log.Printf("predicting on neural net")

	var correct int
	for i := range teX {
		result := nn.Predict(teX[i])
		if teY[i][result[0]] > 0 {
			correct++
		}
	}
	log.Printf("neural net classifier accuracy: %0.1f%% (%d / %d)", percent(correct, len(teY)), correct, len(teY))
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
