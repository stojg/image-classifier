package main

import (
	"log"
	"math/rand"
	"time"
)

type trail struct {
	labels [][]float64
	bestW  *Matrix
	bestB  *Matrix
	bestW2 *Matrix
	bestB2 *Matrix
	log    bool
}

func (t *trail) Train(inputX [][]float64, y [][]byte, numEpocs int) {

	const (
		reg      = 1e-3
		stepSize = 0.01
	)

	rand.Seed(time.Now().UTC().UnixNano())

	dataX := make([]float64, 0)
	for i := range inputX {
		dataX = append(dataX, inputX[i]...)
	}
	x := NewMatrixF(dataX, len(inputX), len(inputX[0]))

	inputNeurons := len(inputX[0])
	hiddenNeurons := 100
	outputNeurons := len(y[0]) // number of classes

	// initialize parameters randomly
	W := NewRandomMatrix(inputNeurons, hiddenNeurons).ScalarMul(0.01)
	b := NewZerosMatrix(1, hiddenNeurons)

	W2 := NewRandomMatrix(hiddenNeurons, outputNeurons).ScalarMul(0.01)
	b2 := NewZerosMatrix(1, outputNeurons)

	var hScore *Matrix

	for epoch := 0; epoch < numEpocs; epoch++ {

		// evaluate class scores with a 2-layer Neural Network
		hLayer := x.Dot(W).RowAdd(b).ElementMax(0) // Rectified linear unit (ReLU) activation
		hScore = hLayer.Dot(W2).RowAdd(b2)

		if t.log && epoch%1 == 0 {
			loss := t.loss(hScore, W, W2, y, reg)
			log.Printf("epoch %d: loss: %f, accuracy %.1f%%", epoch+1, loss, t.accuracy(hScore, y)*100)

		}

		dScores := hScore.Clone()
		for row := range y {
			for col := range y[row] {
				if y[row][col] > 0 {
					// by decreasing the correct class probability with one
					// we find the descent in the direction where the
					// loss function is decreasing the most
					dScores.data[row*dScores.cols+col] -= 1 //
					break
				}
			}
		}
		dScores = dScores.ScalarDiv(float64(x.rows))

		// back propagate the gradient to the parameters
		// 1) back propagate into parameters W2 and b2 (output)
		dW2 := hLayer.Transpose().Dot(dScores)
		db2 := dScores.ColSum()
		// 2) next back propagate into hidden layer
		dHidden := dScores.Dot(W2.Transpose())
		// back propagate the ReLU non-linearity
		for i := range dHidden.data {
			if hLayer.data[i] < 0 {
				dHidden.data[i] = 0
			}
		}

		// finally into W,b (input)
		dW := x.Transpose().Dot(dHidden)
		db := dHidden.ColSum()

		// add regularization gradient contribution
		dW2 = dW2.Add(W2.ScalarMul(reg))
		dW = dW.Add(W.ScalarMul(reg))

		// parameter update
		W = W.Add(dW.ScalarMul(-stepSize))
		b = b.Add(db.ScalarMul(-stepSize))
		W2 = W2.Add(dW2.ScalarMul(-stepSize))
		b2 = b2.Add(db2.ScalarMul(-stepSize))
	}
	t.bestW2 = W2
	t.bestB2 = b2
	t.bestW = W
	t.bestB = b

	if t.log {
		loss := t.loss(hScore, W, W2, y, reg)
		log.Printf("result: loss: %f, accuracy %.1f%%", loss, t.accuracy(hScore, y)*100)
	}
}

func (t *trail) Predict(input []float64) []int {
	xTe := NewMatrixF(input, 1, len(input))

	// evaluate class scores with a 2-layer Neural Network
	hiddenLayer := xTe.Dot(t.bestW).RowAdd(t.bestB).ElementMax(0) // ReLU activation
	scores := hiddenLayer.Dot(t.bestW2).RowAdd(t.bestB2)

	return scores.ArgMax()
}

func (t *trail) loss(scores, W, W2 *Matrix, y [][]byte, reg float64) float64 {
	// get unnormalized probabilities
	scoresExp := scores.Clone().ScalarExp()
	// normalize them for each example
	probs := scoresExp.ColDiv(scoresExp.RowSum())
	// We can now query for the log probabilities assigned to the correct classes in each example:
	correctLogProbs := probs.RowFinder(y).ScalarMinusLog()

	// compute the loss: average cross-entropy loss and regularization
	data_loss := correctLogProbs.Sum() / float64(len(y))
	reg_loss := 0.5*reg*W.ElementMul(W).Sum() + 0.5*reg*W2.ElementMul(W2).Sum()
	loss := data_loss + reg_loss

	return loss
}

func (t *trail) accuracy(scores *Matrix, y [][]byte) float64 {
	actual := scores.ArgMax()
	correct := 0.0
	for i := range actual {
		if y[i][actual[i]] > 0 {
			correct += 1
		}
	}
	return correct / float64(len(y))
}
