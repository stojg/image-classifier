package main

import (
	"math"
)

const (
	NormNoop     byte = iota
	NormNumeric       // Gaussian normalization generally between -10 and +10
	NormCategory      //
)

type Normaliser struct {
}

func (n *Normaliser) Transpose(input [][]float64) [][]float64 {
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

func (n *Normaliser) Normalise(input [][]float64, patterns []byte) [][]float64 {

	if len(input[0]) != len(patterns) {
		panic("input must have the same width as the patterns")
	}

	br := n.Transpose(input)
	var resultX [][]float64
	for i := range br {
		switch patterns[i] {
		case NormNumeric:
			resultX = append(resultX, n.numeric(br[i]))
		case NormCategory:
			cats := n.categorise(br[i])
			for _, cat := range cats {
				resultX = append(resultX, cat)
			}
		default:
			resultX = append(resultX, br[i])
		}
	}
	return n.Transpose(resultX)
}

func (n *Normaliser) categorise(rows []float64) [][]float64 {
	categories := n.removeDuplicates(rows)
	matrix := make([][]float64, len(rows))
	for i := 0; i < len(rows); i++ {
		matrix[i] = make([]float64, len(categories))
		for j := 0; j < len(matrix[i]); j++ {
			if rows[i] == categories[j] {
				matrix[i][j] = 1.0
			} else {
				matrix[i][j] = -1.0
			}
		}
	}
	return n.Transpose(matrix)
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
	total := 0.0
	for _, number := range numbers {
		total += math.Pow(number-mean, 2)
	}
	variance := total / float64(len(numbers)-1)
	return math.Sqrt(variance)
}

func (n *Normaliser) removeDuplicates(a []float64) []float64 {
	result := []float64{}
	seen := map[float64]float64{}
	for _, val := range a {
		if _, ok := seen[val]; !ok {
			result = append(result, val)
			seen[val] = val
		}
	}
	return result
}
