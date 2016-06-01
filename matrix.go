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

func NewRandomMatrix(rows, cols int) *Matrix {
	t := make([]float64, rows*cols)
	for i := range t {
		t[i] = rand.NormFloat64()
	}
	return NewMatrixF(t, rows, cols)
}

func NewZeros(rows, cols int) *Matrix {
	t := make([]float64, cols*rows)
	return NewMatrixF(t, rows, cols)
}

func NewOnes(rows, cols int) *Matrix {
	t := make([]float64, cols*rows)
	for i := range t {
		t[i] = 1
	}
	return NewMatrixF(t, rows, cols)
}

func (A *Matrix) At(row, col int) float64 {
	return A.Data[row*A.Cols+col]
}

// SDot is Dot implementation that is faster on very small matrices
func (A *Matrix) SDot(B *Matrix) *Matrix {
	if A.Cols != B.Rows {
		panic(fmt.Sprintf("matrix.Mul() A (%d X %d) * B (%d X %d)", A.Rows, A.Cols, B.Rows, B.Cols))
	}

	result := make([]float64, A.Rows*B.Cols)
	row := make([]float64, A.Cols)

	for r := 0; r < A.Rows; r++ {
		for col := range row {
			row[col] = A.Data[r*A.Cols+col]
		}
		for c := 0; c < B.Cols; c++ {
			var v float64
			for i, e := range row {
				v += e * B.Data[i*B.Cols+c]
			}
			result[r*B.Cols+c] = v
		}
	}
	return NewMatrixF(result, A.Rows, B.Cols)
}

func (A *Matrix) Dot(B *Matrix) *Matrix {
	if A.Cols != B.Rows {
		panic(fmt.Sprintf("matrix.Mul() A (%d X %d) * B (%d X %d)", A.Rows, A.Cols, B.Rows, B.Cols))
	}

	result := make([]float64, A.Rows*B.Cols)

	in := make(chan int)
	quit := make(chan bool)

	dotRowCol := func() {
		for {
			select {
			case i := <-in:
				sums := make([]float64, B.Cols)
				for k := 0; k < A.Cols; k++ {
					for j := 0; j < B.Cols; j++ {
						sums[j] += A.Data[i*A.Cols+k] * B.Data[k*B.Cols+j]
					}
				}
				for j := 0; j < B.Cols; j++ {
					result[i*B.Cols+j] = sums[j]
				}
			case <-quit:
				return
			}
		}
	}

	threads := 2

	for i := 0; i < threads; i++ {
		go dotRowCol()
	}

	for i := 0; i < A.Rows; i++ {
		in <- i
	}

	for i := 0; i < threads; i++ {
		quit <- true
	}

	return NewMatrixF(result, A.Rows, B.Cols)
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

func (A *Matrix) ElementSquare() *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = A.Data[i] * A.Data[i]
	}
	return NewMatrixF(res, A.Rows, A.Cols)
}

func (A *Matrix) ElementLog() *Matrix {
	res := make([]float64, len(A.Data))
	for i := range res {
		res[i] = math.Log(A.Data[i])
	}
	return NewMatrixF(res, A.Rows, A.Cols)
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

func (A *Matrix) Clone() *Matrix {
	clonedData := make([]float64, len(A.Data))
	copy(clonedData, A.Data)
	return NewMatrixF(clonedData, A.Rows, A.Cols)
}

func (A *Matrix) AddBias() *Matrix {
	res := make([]float64, A.Rows*A.Cols+A.Rows)

	for row := 0; row < A.Rows; row++ {
		stride := row*A.Cols + 1
		fromStride := stride - 1
		length := A.Cols
		copy(res[stride+row:stride+length+row], A.Data[fromStride:fromStride+length])
		res[row*A.Cols+row] = 1
	}
	return NewMatrixF(res, A.Rows, A.Cols+1)
}

func (A *Matrix) RemoveBias() *Matrix {
	res := make([]float64, A.Rows*A.Cols-A.Rows)

	for row := 0; row < A.Rows; row++ {
		stride := row * A.Cols
		fromStride := stride + 1
		length := A.Cols - 1
		copy(res[stride-row:stride+length-row], A.Data[fromStride:fromStride+length])
	}
	return NewMatrixF(res, A.Rows, A.Cols-1)
}

func (A *Matrix) ZeroBias() *Matrix {
	res := A.Clone()
	for i := 0; i < A.Rows; i++ {
		res.Data[i*A.Cols] = 0
	}
	return res
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
	fmt.Printf("--- %d X %d ---\n", A.Rows, A.Cols)
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
