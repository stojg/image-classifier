package main

import (
	"math"
)

type Normaliser struct {
}

func (n *Normaliser) Normalise(input [][]float64) [][]float64 {
	br := n.transpose(input)
	var resultX [][]float64
	for i := range br {
		resultX = append(resultX, n.numeric(br[i]))
	}
	return n.transpose(resultX)
}

func (n *Normaliser) transpose(input [][]float64) [][]float64 {
	b := make([][]float64, len(input[0]))
	for i := 0; i < len(input[0]); i++ {
		b[i] = make([]float64, len(input))
	}
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[0]); j++ {
			b[j][i] = input[i][j]
		}
	}
	return b
}

func (n *Normaliser) numeric(values []float64) []float64 {
	var result []float64

	mean := n.sum(values) / float64(len(values))

	stdDev := n.stdDev(values, mean)
	for k := range values {
		result = append(result, (values[k]-mean)/stdDev)
	}
	return result
}

func (n *Normaliser) sum(numbers []float64) (total float64) {
	for _, x := range numbers {
		total += x
	}
	return total
}

func (n *Normaliser) stdDev(numbers []float64, mean float64) float64 {
	if len(numbers) == 1 {
		return 1.0
	}
	total := 0.0
	for _, number := range numbers {
		total += math.Pow(number-mean, 2)
	}
	var variance float64
	variance = total / float64(len(numbers)-1)
	if variance == 0 {
		return 1.0
	}

	return math.Sqrt(variance)
}
