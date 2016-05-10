package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Vector struct {
	cols int
	rows int
	data []float64
}

type Matrix struct {
	rows int
	cols int
	data []float64
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
		data: flattened,
	}
}

func NewRandomMatrix(rows, cols int) *Matrix {
	t := make([]float64, rows*cols)
	for i := range t {
		t[i] = rand.NormFloat64()
	}
	return NewMatrixF(t, rows, cols)
}

func NewZerosMatrix(rows, cols int) *Matrix {
	t := make([]float64, cols*rows)
	return NewMatrixF(t, rows, cols)
}

func NewOnesMatrix(rows, cols int) *Matrix {
	t := make([]float64, cols*rows)
	for i := range t {
		t[i] = 1
	}
	return NewMatrixF(t, rows, cols)
}

func NewMatrixF(m []float64, rows, cols int) *Matrix {

	if len(m) != rows*cols {
		panic(fmt.Sprintf("rows (%d) and cols (%d) dont match up with data length (%d)", rows, cols, len(m)))
	}
	return &Matrix{
		rows: rows,
		cols: cols,
		data: m,
	}
}

func NewVector(v []float64) *Vector {
	return &Vector{
		rows: len(v),
		cols: 1,
		data: v,
	}
}

func (V *Vector) Len() int {
	return V.rows
}

func (V *Vector) Transpose() *Matrix {
	return NewMatrixF(V.data, V.rows, 1)
}

func (A *Matrix) Dim() (int, int) {
	return A.rows, A.cols
}

func (A *Matrix) Idx(v float64) (int, int) {
	for i := range A.data {
		if A.data[i] == v {
			row := math.Floor(float64(i) / float64(A.cols))
			col := math.Floor(float64(i) / float64(A.rows))
			return int(row), int(col)
		}
	}
	return 0, 0
}

func (A *Matrix) At(row, col int) float64 {
	return A.data[row*A.cols+col]
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
			row[aCol] = A.data[r*aCols+aCol]
		}
		var v float64
		for i, e := range row {
			v += e * B.data[i*bCols]
		}
		result[r*bCols] = v
	}
	return NewMatrixF(result, aRows, bCols)
}

func (A *Matrix) Dot(B *Matrix) *Matrix {
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
			row[col] = A.data[r*aCols+col]
		}
		for c := 0; c < bCols; c++ {
			var v float64
			for i, e := range row {
				v += e * B.data[i*bCols+c]
			}
			result[r*bCols+c] = v
		}
	}
	return NewMatrixF(result, aRows, bCols)
}

func (A *Matrix) RowAdd(B *Matrix) *Matrix {
	if A.cols != B.cols {
		panic("dont have the same num of cols!")
	}
	res := make([]float64, len(A.data))
	for i := range A.data {
		for j := range B.data {
			res[i] = A.data[i] + B.data[j]
		}
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) RowSum() *Matrix {
	res := make([]float64, A.rows)
	for row := 0; row < A.rows; row++ {
		sum := 0.0
		for col := 0; col < A.cols; col++ {
			sum += A.data[row*A.cols+col]
		}
		res[row] = sum
	}
	return NewMatrixF(res, A.rows, 1)
}

func (A *Matrix) ColSum() *Matrix {
	res := make([]float64, A.cols)
	for row := 0; row < A.rows; row++ {
		for col := 0; col < A.cols; col++ {
			res[col] += A.data[row*A.cols+col]
		}
	}
	return NewMatrixF(res, 1, A.cols)
}

func (A *Matrix) ColDiv(B *Matrix) *Matrix {
	if A.rows != B.rows {
		panic("A and B dont have the same # of cols")
	}

	if B.cols != 1 {
		panic("B.cols must be 1")
	}

	res := make([]float64, len(A.data))
	for row, divider := range B.data {
		for col := 0; col < A.cols; col++ {
			res[row*A.cols+col] = A.data[row*A.cols+col] / divider
		}
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) ArgMax() []int {
	res := make([]int, A.rows)
	for row := 0; row < A.rows; row++ {
		highest := math.Inf(-1)
		for col := 0; col < A.cols; col++ {
			if A.data[row*A.cols+col] > highest {
				res[row] = int(col)
				highest = A.data[row*A.cols+col]
			}
		}
	}
	return res
}

func (A *Matrix) Add(B *Matrix) *Matrix {
	if A.cols != B.cols || A.rows != B.rows {
		panic(fmt.Sprintf("matrix.Add() matrices must be the same size, A %dX%d, B %dX%d", A.rows, A.cols, B.rows, B.cols))
	}
	res := make([]float64, len(A.data))
	for i := range A.data {
		res[i] = A.data[i] + B.data[i]
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) Sub(B *Matrix) *Matrix {
	if A.cols != B.cols || A.rows != B.rows {
		panic(fmt.Sprintf("matrix.Add() matrices must be the same size"))
	}
	res := make([]float64, len(A.data))
	for i := range A.data {
		res[i] = A.data[i] - B.data[i]
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) Equals(B *Matrix) bool {
	if A.rows != B.rows || A.cols != B.cols {
		return false
	}
	for i := range A.data {
		if A.data[i] != B.data[i] {
			return false
		}
	}
	return true
}

func (A *Matrix) Max() float64 {
	max := math.Inf(-1)
	for _, val := range A.data {
		if val > max {
			max = val
		}
	}
	return max
}

func (A *Matrix) Min() float64 {
	max := math.Inf(1)
	for _, val := range A.data {
		if val < max {
			max = val
		}
	}
	return max
}

func (A *Matrix) Sum() float64 {
	var sum float64
	for _, val := range A.data {
		sum += val
	}
	return sum
}

func (A *Matrix) ElementMul(B *Matrix) *Matrix {
	res := make([]float64, len(A.data))
	for i := range A.data {
		res[i] = A.data[i] * B.data[i]
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) RowFinder(y []int) *Matrix {
	rows := len(y)
	data := make([]float64, rows)
	for t := 0; t < rows; t++ {
		data[t] = A.At(t, y[t])
	}
	return NewMatrixF(data, rows, 1)
}

func (A *Matrix) AsIntSlice() []int {
	d := make([]int, len(A.data))
	for i, val := range A.data {
		d[i] = int(val)
	}
	return d
}

func (A *Matrix) ScalarSub(v float64) *Matrix {
	res := make([]float64, len(A.data))
	for i := range res {
		res[i] = A.data[i] - v
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) ScalarMul(val float64) *Matrix {
	res := make([]float64, len(A.data))
	for i := range res {
		res[i] = A.data[i] * val
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) ScalarDiv(val float64) *Matrix {
	res := make([]float64, len(A.data))
	for i := range res {
		res[i] = A.data[i] / val
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) ScalarExp() *Matrix {
	res := make([]float64, len(A.data))
	for i := range res {
		res[i] = math.Exp(A.data[i])
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) ScalarMinusLog() *Matrix {
	res := make([]float64, len(A.data))
	for i := range res {
		res[i] = -math.Log(A.data[i])
	}
	return NewMatrixF(res, A.rows, A.cols)
}

func (A *Matrix) Clone() *Matrix {
	return NewMatrixF(A.data, A.rows, A.cols)
}

func (A *Matrix) AbsSum() float64 {
	var sum float64
	for i := range A.data {
		if A.data[i] < 0 {
			sum += -A.data[i]
		} else {
			sum += A.data[i]
		}
	}
	return sum
}

func (A *Matrix) Print() {
	fmt.Printf("--- %d X %d ---\n[", A.rows, A.cols)

	for i := range A.data {
		fmt.Printf("\t%0.2f\t", A.data[i])
		if i%A.cols == A.cols-1 {
			if i+1 == len(A.data) {
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
	t := make([]float64, len(A.data))
	for row := 0; row < A.rows; row++ {
		for col := 0; col < A.cols; col++ {
			t[col*A.rows+row] = A.data[row*A.cols+col]
		}
	}
	return NewMatrixF(t, A.cols, A.rows)
}
