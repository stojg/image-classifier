package main

import "fmt"

type Vector struct {
	cols int
	rows int
	raw  []float64
}

type Matrix struct {
	rows int
	cols int
	raw  []float64
}

func NewMatrix(m [][]float64) *Matrix {
	// flatten
	flattened := make([]float64, 0)
	for i := range m {
		flattened = append(flattened, m[i]...)
	}

	return &Matrix{
		rows: len(m),
		cols: len(m[0]),
		raw:  flattened,
	}
}

func NewMatrixF(m []float64, rows, cols int) *Matrix {
	return &Matrix{
		rows: rows,
		cols: cols,
		raw:  m,
	}
}

func NewVector(v []float64) *Vector {
	return &Vector{
		rows: len(v),
		cols: 1,
		raw:  v,
	}
}

func (V *Vector) Len() int {
	return V.rows
}

func (V *Vector) Transpose() *Matrix {
	return NewMatrixF(V.raw, V.rows, 1)
}

func (A *Matrix) Dim() (int, int) {
	return A.rows, A.cols
}

func (A *Matrix) MulVec(B *Vector) *Matrix {
	if A.cols != B.rows {
		panic(fmt.Sprintf("matrix.MulVec() A.cols (%d) != v.rows (%d)", A.cols, B.rows))
	}

	aRows := A.rows
	aCols := A.cols
	bCols := B.cols

	result := make([]float64, aRows*bCols)
	row := make([]float64, aCols)

	for r := 0; r < aRows; r++ {
		for aCol := range row {
			row[aCol] = A.raw[r*aCols+aCol]
		}
		var v float64
		for i, e := range row {
			v += e * B.raw[i*bCols]
		}
		result[r*bCols] = v
	}
	return NewMatrixF(result, aRows, bCols)
}

func (A *Matrix) Mul(B *Matrix) *Matrix {
	if A.cols != B.rows {
		panic(fmt.Sprintf("matrix.Mul() A.cols (%d) != B.rows (%d)", A.cols, B.rows))
	}

	aRows := A.rows
	aCols := A.cols
	bCols := B.cols

	result := make([]float64, aRows*bCols)
	row := make([]float64, aCols)

	for r := 0; r < aRows; r++ {
		for col := range row {
			row[col] = A.raw[r*aCols+col]
		}
		for c := 0; c < bCols; c++ {
			var v float64
			for i, e := range row {
				v += e * B.raw[i*bCols+c]
			}
			result[r*bCols+c] = v
		}
	}
	return NewMatrixF(result, aRows, bCols)
}

func (A *Matrix) Add(B *Matrix) *Matrix {
	if A.cols != B.cols || A.rows != B.rows {
		panic(fmt.Sprintf("matrix.Add() matrices must be the same size"))
	}
	res := make([]float64, len(A.raw))
	for i := range A.raw {
		res[i] = A.raw[i] + B.raw[i]
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) Equals(B *Matrix) bool {
	if A.rows != B.rows || A.cols != B.cols {
		return false
	}
	for i := range A.raw {
		if A.raw[i] != B.raw[i] {
			return false
		}
	}
	return true
}

func (A *Matrix) Print() {
	fmt.Printf("--- %d X %d ---\n[", A.rows, A.cols)

	for i := range A.raw {
		fmt.Printf("\t%0.1f\t", A.raw[i])
		if i%A.cols == A.cols-1 {
			if i+1 == len(A.raw) {
				fmt.Printf("]")
			} else {
				fmt.Printf("]\n[ ")
			}
		}
	}
	fmt.Printf("\n")
}

func (A *Matrix) slice(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return data
}

func (A *Matrix) Transpose() *Matrix {
	panic("not yet implemented")

}
