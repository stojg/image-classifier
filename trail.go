package main

import (
	"fmt"
	"math/rand"
	"time"
)

type trail struct {
	labels [][]float64
	bestW  *Matrix
	bestb  *Matrix
}

func (t *trail) Train(input [][][]float64) {

	rand.Seed(time.Now().UTC().UnixNano())
	dataX := make([]float64, 0)

	for i := range input {
		dataX = append(dataX, input[i][0]...)
	}

	dataY := make([]float64, 0)
	for i := range input {
		dataY = append(dataY, input[i][1][0])
	}

	D := len(input[0][0]) // dim
	K := 10               // number of classes

	x := NewMatrixF(dataX, len(input), len(input[0][0]))
	y := NewMatrixF(dataY, len(input), 1)

	// initialize parameters randomly
	W := NewRandomMatrix(D, K).ScalarMul(0.1)
	b := NewZerosMatrix(1, K)

	const reg = 1e-3
	const stepSize = 0.001

	var scores *Matrix

	for i := 0; i < 1000; i++ {

		// evaluate class scores,
		scores = x.Dot(W).RowAdd(b)

		if i%1 == 0 {
			fmt.Printf("%d: ", i)
			t.score(scores, W, y, reg)
		}

		// Computing the Analytic Gradient with Back propagation
		dscores := scores.Clone()
		row := 0
		for _, val := range y.AsIntSlice() {
			dscores.data[row*dscores.cols+val] = dscores.data[row*dscores.cols+val] - 1
			row++
		}
		dscores = dscores.ScalarDiv(float64(x.rows))

		// back propagate into W and b
		dW := x.Transpose().Dot(dscores)
		// sum all the bias differences
		db := dscores.ColSum()
		dW = dW.Add(W.ScalarMul(reg))

		// parameter update
		W = W.Add(dW.ScalarMul(-stepSize))
		b = b.Add(db.ScalarMul(-stepSize))
	}

	t.score(scores, W, y, reg)

	t.bestW = W
	t.bestb = b
}

func (t *trail) Predict(input []float64) []int {
	xTe := NewMatrixF(input, 1, len(input))
	scores := xTe.Dot(t.bestW).RowAdd(t.bestb)
	return scores.ArgMax()
}

func (t *trail) score(scores *Matrix, W *Matrix, y *Matrix, reg float64) {
	scoresExp := scores.ScalarExp()
	probs := scoresExp.ColDiv(scoresExp.RowSum())
	correctLogProbs := probs.RowFinder(y.AsIntSlice()).ScalarMinusLog()

	// compute the loss: average cross-entropy loss and regularization
	data_loss := correctLogProbs.Sum() / float64(y.rows)
	reg_loss := 0.5 * reg * W.ElementMul(W).Sum()
	loss := data_loss + reg_loss

	fmt.Printf("loss: %.2f ", loss)
	fmt.Printf("accuracy: %.2f\n", comp(y.AsIntSlice(), scores.ArgMax()))
}

func comp(a, b []int) float64 {
	correct := 0.0
	for i, val := range a {
		if b[i] == val {
			correct += 1
		}
	}
	return correct / float64(len(a))

}
