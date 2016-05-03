package main

import (
	"fmt"
	"log"
	"math"
)

type NearestNeighbour struct {
	data   []CIFAR10Image
	labels []byte
}

func (c *NearestNeighbour) Train(X []CIFAR10Image, y []byte) {
	c.data = X
	c.labels = y
}

func AbsSub(a, b []byte) int {
	var result int
	for k := range a {
		diff := int(a[k]) - int(b[k])
		if diff < 0 {
			diff = -diff
		}
		result += diff
	}
	return result
}

func (c *NearestNeighbour) Predict(testSet []CIFAR10Image) []byte {
	num_test := len(testSet)
	prediction := make([]byte, num_test)
	for i := range testSet {
		lowestScore := math.MaxInt32
		for j := range c.data {
			result := AbsSub(c.data[j].data, testSet[i].data)
			if result < lowestScore {
				lowestScore = result
				prediction[i] = c.data[j].label
			}
		}
		if i > 20 {
			return prediction
		}
		fmt.Printf(".")
	}
	return prediction
	return c.labels[:10000]
}

// compares two slices with the compare func
func mean(xS, yS []byte, compareFunc func(x, y byte) bool) float32 {
	count := 0
	for i := range xS {
		if compareFunc(xS[i], yS[i]) {
			count++
		}
	}
	return float32(count) / float32(len(xS))
}

func main() {
	trainingSet, trainingLabels := loadCIFAR10("data/data_batch_*")
	testSet, testLabels := loadCIFAR10("data/test_batch.bin")
	nn := &NearestNeighbour{}
	nn.Train(trainingSet, trainingLabels)
	Yte_predict := nn.Predict(testSet)
	avg := mean(Yte_predict, testLabels, func(x, y byte) bool {
		return x == y
	})
	log.Printf("accuracy: %f", avg)
}

type Trainer interface {
	Train(trainingImages []byte, labels []byte)
}

type Predictor interface {
	Predict(testData []byte) (testLabels []byte)
}
