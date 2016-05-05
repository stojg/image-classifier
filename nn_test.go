package main

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"testing"
)

func getRandomData(n int, size int) [][][]float64 {
	r := rand.New(rand.NewSource(99))

	trainingData := make([][][]float64, n)
	for i := 0; i < n; i++ {
		trainingData[i] = make([][]float64, 2)
		trainingData[i][0] = make([]float64, size)
		for j := 0; j < size; j++ {
			trainingData[i][0][j] = float64(r.Intn(255))
		}
		trainingData[i][1] = make([]float64, 1)
		trainingData[i][1][0] = float64(r.Intn(255))
	}
	return trainingData
}

func TestNearestNeighbour(t *testing.T) {

	trainingData := getRandomData(2, 32)

	var testData [][]float64
	for idx := range trainingData {
		testData = append(testData, trainingData[idx][0])
	}

	n := &NearestNeighbour{}
	n.Train(trainingData)
	prediction := n.Predict(testData)

	t.Logf("%v", testData)
	if prediction.At(0, 0) != trainingData[0][1][0] {
		t.Errorf("Expected the nn classifier to find exact match with same training and test data")
	}

	if prediction.At(1, 0) != trainingData[1][1][0] {
		t.Errorf("Expected the nn classifier to find exact match with same training and test data")
	}
}

var result *mat64.Dense

func BenchmarkNearestNeighbour(b *testing.B) {

	trainingData := getRandomData(1, 1024*3)
	var testData [][]float64
	for idx := range trainingData {
		testData = append(testData, trainingData[idx][0])
	}
	n := &NearestNeighbour{}
	n.Train(trainingData)
	var r *mat64.Dense

	for i := 0; i < b.N; i++ {
		r = n.Predict(testData)
	}
	result = r
}
