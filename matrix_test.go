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

func TestMatrixAt(t *testing.T) {
	m := NewMatrix([][]float64{
		[]float64{0, 1},
		[]float64{2, 3},
	})

	if m.At(0, 0) != 0 {
		t.Errorf("fail when trying to get 0, got %.1f", m.At(0, 0))
	}
	if m.At(0, 1) != 1 {
		t.Errorf("fail when trying to get 1, got %.1f", m.At(0, 1))
	}
	if m.At(1, 0) != 2 {
		t.Errorf("fail when trying to get 2, got %.1f", m.At(1, 0))
	}
	if m.At(1, 1) != 3 {
		t.Errorf("fail when trying to get 3, got %.1f", m.At(1, 1))
	}
}

func TestMatrixMax(t *testing.T) {
	A := NewMatrixF([]float64{0, 1, 2, 3, 4, 5, 4, 3, 2, 1}, 5, 2)

	if A.Max() != 5 {
		t.Errorf("could not find max in Matrix, got %f", A.Max())
	}
}

func TestMatrixMin(t *testing.T) {
	A := NewMatrixF([]float64{5, 4, 3, 2, 1, 0, -4, 3, 1, 56}, 5, 2)

	if A.Min() != -4 {
		t.Errorf("could not find max in Matrix, got %f", A.Min())
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
		actual := A.Dot(B)
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
		actual = A.Dot(B)
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

func TestMatrixSub(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	B := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	expected := NewMatrix([][]float64{
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	})
	actual := A.Sub(B)
	if !actual.Equals(expected) {
		t.Errorf("actual is not the same as expected")
		actual.Print()
		expected.Print()
	}
}

func TestMatrixSum(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})
	expected := 21.0

	if A.Sum() != expected {
		t.Errorf("expected %f, got %f", expected, A.Sum())
	}

}

func TestSubScalar(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	})

	expected := NewMatrix([][]float64{
		[]float64{-4, -3, -2},
		[]float64{-1, 0, 1},
	})

	if !A.ScalarSub(5).Equals(expected) {
		t.Errorf("Matrix.SubScalar() failed")
		A.ScalarSub(5).Print()
	}
}

func TestMatrixTranspose(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{1, 2},
		[]float64{3, 4},
		[]float64{5, 6},
	})

	expected := NewMatrix([][]float64{
		[]float64{1, 3, 5},
		[]float64{2, 4, 6},
	})

	actual := A.Transpose()

	if !actual.Equals(expected) {
		t.Errorf("not correct")
		A.Print()
		actual.Print()
	}
}

func TestArgMax(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{3, 9, 6},
		[]float64{10, 8, 16},
		[]float64{9, 8, 6},
	})

	expected := []int{1, 2, 0}

	if A.ArgMax()[0] != expected[0] {
		t.Errorf("Argmax fail #1")
	}
	if A.ArgMax()[1] != expected[1] {
		t.Errorf("Argmax fail #2")
	}
	if A.ArgMax()[2] != expected[2] {
		t.Errorf("Argmax fail #3")
	}
}

func TestMatrixColDiv(t *testing.T) {
	A := NewMatrix([][]float64{
		[]float64{3, 6, 9},
		[]float64{10, 8, 16},
	})
	B := NewMatrix([][]float64{
		[]float64{3},
		[]float64{2},
	})

	expected := NewMatrix([][]float64{
		[]float64{1, 2, 3},
		[]float64{5, 4, 8},
	})

	actual := A.ColDiv(B)
	if !actual.Equals(expected) {
		t.Errorf("not the expected")
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
