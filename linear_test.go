package main

import (
	"testing"
)

func TestStuff(t *testing.T) {

	test := NewVector([]float64{1, 2, 3})

	weights := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	})

	b := NewVector([]float64{1, 1, 1})

	expected := NewMatrix([][]float64{
		[]float64{1*1 + 2*2 + 3*3 + 1}, // score for class 1
		[]float64{4*1 + 5*2 + 6*3 + 1}, // score for class 2
		[]float64{7*1 + 8*2 + 9*3 + 1}, // score for class 3
	})

	actual := linear(test, weights, b)

	if !actual.Equals(expected) {
		t.Errorf("not the same")
		actual.Print()
		expected.Print()
	}

}

var linResult *Matrix

func BenchmarkStuff(b *testing.B) {
	test := NewVector([]float64{1, 2, 3})

	weights := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
		[]float64{7, 8, 9},
		[]float64{7, 8, 9},
		[]float64{7, 8, 9},
		[]float64{7, 8, 9},
		[]float64{7, 8, 9},
	})

	bias := NewVector([]float64{1, 1, 1, 1, 1, 1, 1, 1})

	var actual *Matrix
	for i := 0; i < b.N; i++ {
		actual = linear(test, weights, bias)
	}
	linResult = actual
}
