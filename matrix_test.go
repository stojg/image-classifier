package main

import (
	"testing"
)

// benchmark var defined at package level to ensure that the compiler doesn't optimize them
var bResult *Matrix

func TestNewMatrix(t *testing.T) {
	input := [][]float64{
		[]float64{1, 4, 7, 345},
		[]float64{2, 5, 8, 345},
		[]float64{3, 6, 9, 214},
	}
	m := NewMatrix(input)
	rows, cols := m.Dim()
	if rows != 3 && cols != 4 {
		t.Errorf("expected correct dims")
	}

}

func TestNewVector(t *testing.T) {
	input := []float64{1, 4, 7, 345}
	m := NewVector(input)
	rows := m.Len()
	if rows != 4 {
		t.Errorf("expected correct dims")
	}

}

func TestMatrixMulVector(t *testing.T) {
	M := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	v := NewVector([]float64{1, 2, 3})

	expected := NewMatrix([][]float64{
		[]float64{1*1 + 2*2 + 3*3},
		[]float64{4*1 + 5*2 + 6*3},
	})

	actual := M.MulVec(v)

	if !actual.Equals(expected) {
		t.Errorf("Not the same")
	}
}

func BenchmarkMatrixMulVector(b *testing.B) {
	M := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})
	v := NewVector([]float64{1, 2, 3})
	var actual *Matrix
	for i := 0; i < b.N; i++ {
		actual = M.MulVec(v)
	}
	bResult = actual
}

var matrixMulTestTable = [][][][]float64{
	{
		[][]float64{
			[]float64{1, 2, 3},
			[]float64{4, 5, 6},
		},
		[][]float64{
			[]float64{7, 8},
			[]float64{9, 10},
			[]float64{11, 12},
		},
		[][]float64{
			[]float64{58, 64},
			[]float64{139, 154},
		},
	},
	{
		[][]float64{
			[]float64{3, 4, 2},
		},
		[][]float64{
			[]float64{13, 9, 7, 15},
			[]float64{8, 7, 4, 6},
			[]float64{6, 4, 0, 3},
		},
		[][]float64{
			[]float64{83, 63, 37, 75},
		},
	},
}

func TestMatrixMul(t *testing.T) {
	for _, test := range matrixMulTestTable {
		A := NewMatrix(test[0])
		B := NewMatrix(test[1])
		expected := NewMatrix(test[2])
		actual := A.Mul(B)
		if !actual.Equals(expected) {
			t.Errorf("actual is not the same as expected")
			actual.Print()
			expected.Print()
		}
	}
}

func BenchmarkMatrixMul(b *testing.B) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	B := NewMatrix([][]float64{
		[]float64{7, 8},
		[]float64{9, 10},
		[]float64{11, 12},
	})
	var actual *Matrix
	for i := 0; i < b.N; i++ {
		actual = A.Mul(B)
	}
	bResult = actual
}

func TestMatrixAdd(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	B := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	expected := NewMatrix([][]float64{
		[]float64{2, 4, 6},
		[]float64{8, 10, 12},
	})
	actual := A.Add(B)
	if !actual.Equals(expected) {
		t.Errorf("actual is not the same as expected")
		actual.Print()
		expected.Print()
	}

}

func BenchmarkMatrixAdd(b *testing.B) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	B := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})
	var actual *Matrix
	for i := 0; i < b.N; i++ {
		actual = A.Add(B)
	}
	bResult = actual
}
