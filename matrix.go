package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func NewMatrix(m [][]float64) *Matrix {
	// flatten
	flattened := make([]float64, 0)
	for i := range m {
		flattened = append(flattened, m[i]...)
	}

	return &Matrix{
		Rows: len(m),
		Cols: len(m[0]),
		Data: flattened,
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
		Rows: rows,
		Cols: cols,
		Data: m,
	}
}

func (A *Matrix) Dim() (int, int) {
	return A.Rows, A.Cols
}

func (A *Matrix) Idx(v float64) (int, int) {
	for i := range A.Data {
		if A.Data[i] == v {
			row := math.Floor(float64(i) / float64(A.Cols))
			col := math.Floor(float64(i) / float64(A.Rows))
			return int(row), int(col)
		}
	}
	return 0, 0
}

func (A *Matrix) At(row, col int) float64 {
	return A.Data[row*A.Cols+col]
}

func (A *Matrix) Dot(B *Matrix) *Matrix {
	if A.Cols != B.Rows {
		panic(fmt.Sprintf("matrix.Mul() A.cols (%d) != B.rows (%d)", A.Cols, B.Rows))
	}

	aRows := A.Rows
	aCols := A.Cols
	bCols := B.Cols

	result := make([]float64, aRows*bCols)
	row := make([]float64, aCols)

	for r := 0; r < aRows; r++ {
		for col := range row {
			row[col] = A.Data[r*aCols+col]
		}
		for c := 0; c < bCols; c++ {
			var v float64
			for i, e := range row {
				v += e * B.Data[i*bCols+c]
			}
			result[r*bCols+c] = v
		}
	}
	return NewMatrixF(result, aRows, bCols)
}

func (A *Matrix) RowAdd(B *Matrix) *Matrix {
	if A.Cols != B.Cols {
		panic("dont have the same num of cols!")
	}
	res := make([]float64, len(A.Data))
	for i := range A.Data {
		for j := range B.Data {
			res[i] = A.Data[i] + B.Data[j]
		}
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) RowSum() *Matrix {
	res := make([]float64, A.Rows)
	for row := 0; row < A.Rows; row++ {
		sum := 0.0
		for col := 0; col < A.Cols; col++ {
			sum += A.Data[row*A.Cols+col]
		}
		res[row] = sum
	}
	return NewMatrixF(res, A.Rows, 1)
}

func (A *Matrix) ColSum() *Matrix {
	res := make([]float64, A.Cols)
	for row := 0; row < A.Rows; row++ {
		for col := 0; col < A.Cols; col++ {
			res[col] += A.Data[row*A.Cols+col]
		}
	}
	return NewMatrixF(res, 1, A.Cols)
}

func (A *Matrix) ColSumOld() *Matrix {
	res := make([]float64, A.Cols)
	for row := 0; row < A.Rows; row++ {
		for col := 0; col < A.Cols; col++ {
			res[col] += A.Data[row*A.Cols+col]
		}
	}
	return NewMatrixF(res, 1, A.Cols)
}

func (A *Matrix) ColDiv(B *Matrix) *Matrix {
	if A.Rows != B.Rows {
		panic("A and B dont have the same # of cols")
	}

	if B.Cols != 1 {
		panic("B.cols must be 1")
	}

	res := make([]float64, len(A.Data))
	for row, divider := range B.Data {
		for col := 0; col < A.Cols; col++ {
			res[row*A.Cols+col] = A.Data[row*A.Cols+col] / divider
		}
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ArgMax() []int {
	res := make([]int, A.Rows)
	for row := 0; row < A.Rows; row++ {
		highest := math.Inf(-1)
		for col := 0; col < A.Cols; col++ {
			if A.Data[row*A.Cols+col] > highest {
				res[row] = int(col)
				highest = A.Data[row*A.Cols+col]
			}
		}
	}
	return res
}

func (A *Matrix) Add(B *Matrix) *Matrix {
	if A.Cols != B.Cols || A.Rows != B.Rows {
		panic(fmt.Sprintf("matrix.Add() matrices must be the same size, A %dX%d, B %dX%d", A.Rows, A.Cols, B.Rows, B.Cols))
	}
	res := make([]float64, len(A.Data))
	for i := range A.Data {
		res[i] = A.Data[i] + B.Data[i]
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) Sub(B *Matrix) *Matrix {
	if A.Cols != B.Cols || A.Rows != B.Rows {
		panic(fmt.Sprintf("matrix.Add() matrices must be the same size"))
	}
	res := make([]float64, len(A.Data))
	for i := range A.Data {
		res[i] = A.Data[i] - B.Data[i]
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) Equals(B *Matrix) bool {
	if A.Rows != B.Rows || A.Cols != B.Cols {
		return false
	}
	for i := range A.Data {
		if A.Data[i] != B.Data[i] {
			return false
		}
	}
	return true
}

func (A *Matrix) Max() float64 {
	max := math.Inf(-1)
	for _, val := range A.Data {
		if val > max {
			max = val
		}
	}
	return max
}

func (A *Matrix) ElementMax(max float64) *Matrix {
	res := make([]float64, len(A.Data))
	for i := range A.Data {
		res[i] = math.Max(max, A.Data[i])
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) Min() float64 {
	max := math.Inf(1)
	for _, val := range A.Data {
		if val < max {
			max = val
		}
	}
	return max
}

func (A *Matrix) Sum() float64 {
	var sum float64
	for _, val := range A.Data {
		sum += val
	}
	return sum
}

func (A *Matrix) ElementMul(B *Matrix) *Matrix {
	res := make([]float64, len(A.Data))
	for i := range A.Data {
		res[i] = A.Data[i] * B.Data[i]
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) RowFinder(y [][]byte) *Matrix {

	rows := len(y)
	result := make([]float64, rows)
	for row := range y {
		for col, val := range y[row] {
			if val > 0 {
				result[row] = A.At(row, col)
				continue
			} else {
			}
		}
	}
	return NewMatrixF(result, rows, 1)
}

func (A *Matrix) AsIntSlice() []int {
	d := make([]int, len(A.Data))
	for i, val := range A.Data {
		d[i] = int(val)
	}
	return d
}

func (A *Matrix) ScalarSub(v float64) *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = A.Data[i] - v
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ScalarMul(val float64) *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = A.Data[i] * val
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ScalarDiv(val float64) *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = A.Data[i] / val
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ScalarExp() *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = math.Exp(A.Data[i])
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ScalarMinusLog() *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = -math.Log(A.Data[i])
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) Clone() *Matrix {
	clonedData := make([]float64, len(A.Data))
	copy(clonedData, A.Data)
	return NewMatrixF(clonedData, A.Rows, A.Cols)
}

func (A *Matrix) AbsSum() float64 {
	var sum float64
	for i := range A.Data {
		if A.Data[i] < 0 {
			sum += -A.Data[i]
		} else {
			sum += A.Data[i]
		}
	}
	return sum
}

func (A *Matrix) Print() {
	fmt.Printf("--- %d X %d ---\n[", A.Rows, A.Cols)

	for i := range A.Data {
		fmt.Printf("\t%0.2f\t", A.Data[i])
		if i%A.Cols == A.Cols-1 {
			if i+1 == len(A.Data) {
				fmt.Printf("]")
			} else {
				fmt.Printf("]\n[ ")
			}
		}
	}
	fmt.Printf("\n")
}

func (A *Matrix) PrintSize() {
	fmt.Printf("--- %d X %d ---\n[", A.Rows, A.Cols)
}

func (A *Matrix) slice(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return data
}

func (A *Matrix) T() *Matrix {
	t := make([]float64, len(A.Data))
	for row := 0; row < A.Rows; row++ {
		for col := 0; col < A.Cols; col++ {
			t[col*A.Rows+row] = A.Data[row*A.Cols+col]
		}
	}
	return NewMatrixF(t, A.Cols, A.Rows)
}
