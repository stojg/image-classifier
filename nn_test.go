package main

import (
	"math/rand"
	"testing"
)

func getRandomData() ([]dValue, []dLabel) {
	r := rand.New(rand.NewSource(99))

	testData := make([]dValue, 1024*3)
	for i := 0; i < 1024*3; i++ {
		testData[i] = dValue(r.Intn(255))
	}

	testLabels := make([]dLabel, 1)
	for i := 0; i < 1; i++ {
		testLabels[i] = dLabel(r.Intn(100))
	}
	return testData, testLabels
}

func TestNeareastNeighbour(t *testing.T) {

	testData, testLabels := getRandomData()

	n := &NearestNeighbour{}
	n.Train(testData, testLabels)
	prediction := n.Predict(testData)

	if !(len(prediction) == 1 && prediction[0] == testLabels[0]) {
		t.Errorf("Expected the nn classifier to find exact match with same training and test data")
	}

}

var result []dLabel

func BenchmarkXxx(b *testing.B) {
	testData, testLabels := getRandomData()

	n := &NearestNeighbour{}
	n.Train(testData, testLabels)

	var r []dLabel
	for i := 0; i < b.N; i++ {
		r = n.Predict(testData)
	}
	result = r
}
