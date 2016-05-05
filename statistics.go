package main

import "github.com/gonum/matrix/mat64"

// compares two slices with the compare func
func labelCompare(predictions *mat64.Dense, actual []float64, compareFunc func(x, y float64) bool) float64 {
	count := 0

	rows, _ := predictions.Dims()
	for i := 0; i < rows; i++ {
		if compareFunc(predictions.At(i, 0), actual[i]) {
			count++
		}
	}
	return float64(count) / float64(rows)
}
