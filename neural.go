package main

import (
	"bitbucket.org/binet/go-gnuplot/pkg/gnuplot"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"
)

type NeuralNet struct {
	labels     [][]float64
	W          *Matrix
	bias       *Matrix
	hiddenW    *Matrix
	hiddenBias *Matrix
	log        bool
	plot       bool
	lossPlot   *gnuplot.Plotter
	accPlot    *gnuplot.Plotter
}

// @todo print time per epoch
// @todo mini batch updates
// @todo add more depth with convnets for image processing
// @todo use all CPU cores
// @todo link with a proper C lib for faster linear algebra (e.g. https://github.com/gonum/blas)
func (t *NeuralNet) Train(xIn [][]float64, yIn [][]byte, numEpochs int) {

	const (
		reg      = 1e-4
		stepSize = 0.02
	)

	log.Printf("Randomising training data")
	for i := range xIn {
		j := rand.Intn(i + 1)
		xIn[i], xIn[j] = xIn[j], xIn[i]
		yIn[i], yIn[j] = yIn[j], yIn[i]
	}

	// setup gnuplots
	t.initPlots()
	defer t.lossPlot.Close()
	defer t.accPlot.Close()

	// use 10% of the test data for prediction, so we can check if we are over fitting the net, this
	// data is not used to train the net.
	x, y, xPred, yPred := t.divide(10, xIn, yIn)

	// bog standard seed of psuedo random generator, set to a fixed number to have a determistic
	// starting point
	rand.Seed(time.Now().UTC().UnixNano())

	inputNeurons := len(xIn[0])
	hiddenNeurons := 100
	outputNeurons := len(y[0])

	// Put up a checkpoint so we can check that the data and loss is set up correctly
	if t.log {
		log.Printf("checkpoint: initial loss should be roughly %f (1 out of %d)", -math.Log(1/float64(outputNeurons)), outputNeurons)
	}

	// initialize parameters (weights) randomly for input -> hidden layer
	Win := NewRandomMatrix(inputNeurons, hiddenNeurons).ScalarMul(0.01)
	biasIn := NewZerosMatrix(1, hiddenNeurons)

	// initialize parameters (weights) randomly for hiddent-> output layer
	WHidden := NewRandomMatrix(hiddenNeurons, outputNeurons).ScalarMul(0.01)
	biasHidden := NewZerosMatrix(1, outputNeurons)

	// keep a running tally of the score between hidden -> output so that we can print
	// it after the epoch loop
	var scores *Matrix

	// these are used so that we can update gnuplots with the data
	var (
		losses      []float64
		accuracies  []float64
		pAccuracies []float64
	)

	// The training loop
	for epoch := 0; epoch < numEpochs; epoch++ {

		// evaluate class scores between input -> hidden
		hiddenScores := t.reluActivation(x.Dot(Win).RowAdd(biasIn))
		// final output layer "scores"
		scores = hiddenScores.Dot(WHidden).RowAdd(biasHidden)

		loss := t.calculateLoss(scores, Win, WHidden, y, reg)
		accuracy := t.accuracy(scores, y) * 100

		// prediction
		prediction := t.reluActivation(xPred.Dot(Win).RowAdd(biasIn))
		predictionAccuracy := t.accuracy(prediction.Dot(WHidden).RowAdd(biasHidden), yPred) * 100

		if t.plot {
			losses = append(losses, loss)
			// accuracy plot
			accuracies = append(accuracies, accuracy)
			pAccuracies = append(pAccuracies, predictionAccuracy)
		}

		if t.log && epoch%25 == 0 {
			log.Printf("epoch %d:  loss: %f,  accuracy %.1f%%", epoch, loss, accuracy)
			if t.plot {
				t.plotProgress(losses, accuracies, pAccuracies)
			}
		}

		dScores := t.GradientDescent(scores, y)

		// back propagate the gradient to the parameters
		// 1) back propagate into the hidden -> output parameters
		dWHidden := hiddenScores.Transpose().Dot(dScores)
		dBiasHidden := dScores.ColSum()
		// 2) next back propagate into input -> hidden parameters
		dHidden := dScores.Dot(WHidden.Transpose())
		// back propagate the ReLU non-linearity
		for i := range dHidden.data {
			if hiddenScores.data[i] < 0 {
				dHidden.data[i] = 0
			}
		}

		// gradient descent on dW and db
		dW := x.Transpose().Dot(dHidden)
		db := dHidden.ColSum()

		// add regularization gradient contribution, basically penalise
		// changes so that we don't overfit
		dWHidden = dWHidden.Add(WHidden.ScalarMul(reg))
		dW = dW.Add(Win.ScalarMul(reg))

		// stochastic parameter update
		Win = Win.Add(dW.ScalarMul(-stepSize))
		biasIn = biasIn.Add(db.ScalarMul(-stepSize))
		WHidden = WHidden.Add(dWHidden.ScalarMul(-stepSize))
		biasHidden = biasHidden.Add(dBiasHidden.ScalarMul(-stepSize))
	}

	t.plotProgress(losses, accuracies, pAccuracies)

	// "close" the gnuplots
	t.lossPlot.Cmd("q")
	t.accPlot.Cmd("q")

	// save the final values so that Predict() can find them
	t.hiddenW = WHidden
	t.hiddenBias = biasHidden
	t.W = Win
	t.bias = biasIn

	if t.log {
		loss := t.calculateLoss(scores, Win, WHidden, y, reg)
		log.Printf("result: loss: %f, accuracy %.1f%%", loss, t.accuracy(scores, y)*100)
	}
}

func (t *NeuralNet) plotProgress(losses, acc, pacc []float64) {

	t.lossPlot.ResetPlot()
	t.lossPlot.PlotX(losses, "loss (cross-entropy)")

	// accuracy plot
	t.accPlot.ResetPlot()
	t.accPlot.PlotX(acc, "test")
	t.accPlot.PlotX(pacc, "validation")

}
func (t *NeuralNet) Predict(input []float64) []int {

	// convert into a matrix
	xTe := NewMatrixF(input, 1, len(input))
	// evaluate class scores with a 2-layer Neural Network
	hiddenLayer := xTe.Dot(t.W).RowAdd(t.bias).ElementMax(0) // ReLU activation
	scores := hiddenLayer.Dot(t.hiddenW).RowAdd(t.hiddenBias)

	return scores.ArgMax()
}

// calculate the loss function with an average cross-entropy
func (t *NeuralNet) calculateLoss(scores, W, W2 *Matrix, y [][]byte, reg float64) float64 {
	// get unnormalized probabilities
	scoresExp := scores.Clone().ScalarExp()
	// normalize them for each example
	probs := scoresExp.ColDiv(scoresExp.RowSum())
	// We can now query for the log probabilities assigned to the correct classes in each example:
	correctLogProbs := probs.RowFinder(y).ScalarMinusLog()
	// compute the loss: average cross-entropy loss and regularization
	// cross-entropy gives us a way to express how different two probability
	// distributions are, see http://colah.github.io/posts/2015-09-Visual-Information/
	data_loss := correctLogProbs.Sum() / float64(len(y))
	reg_loss := 0.5*reg*W.ElementMul(W).Sum() + 0.5*reg*W2.ElementMul(W2).Sum()
	loss := data_loss + reg_loss

	return loss
}

func (t *NeuralNet) GradientDescent(A *Matrix, y [][]byte) *Matrix {
	// find the gradient descent for output
	dScores := A.Clone()
	for row := range y {
		for col := range y[row] {
			if y[row][col] > 0 {
				// by decreasing the hScore[j] that was the correct answer with
				// 1 we find the gradient descent in the direction where the
				// loss function is decreasing the most. But honestly, it's a bit
				// fuzzy for me how the ReLu gradient actually works
				dScores.data[row*dScores.cols+col] -= 1
				break
			}
		}
	}
	// calculate the the average
	return dScores.ScalarDiv(float64(A.rows))
}

//  Rectified Linear unit
func (t *NeuralNet) reluActivation(A *Matrix) *Matrix {
	return A.ElementMax(0)
}

// setup gnuplot for plotting the loss and accuracy (brew install gnuplot)
func (t *NeuralNet) initPlots() {
	var err error

	var lossPlot *gnuplot.Plotter
	if lossPlot, err = gnuplot.NewPlotter("", false, false); err != nil {
		panic(fmt.Sprintf("** err: %v\n", err))
	}

	lossPlot.SetStyle("lines")
	lossPlot.SetXLabel("epoch")
	lossPlot.SetYLabel("loss")

	var accPlot *gnuplot.Plotter

	if accPlot, err = gnuplot.NewPlotter("", false, false); err != nil {
		panic(fmt.Sprintf("** err: %v\n", err))
	}
	accPlot.SetStyle("lines")
	accPlot.SetXLabel("epoch")
	accPlot.SetYLabel("accuracy %")
	accPlot.Cmd("set yrange [0:110]")

	t.lossPlot = lossPlot
	t.accPlot = accPlot
}

// accuracy returns a accuracy between 0.0 - 1.0 on how correct a score (classification)
// matrix is compared to the actual (Y) labels
func (t *NeuralNet) accuracy(scores *Matrix, y [][]byte) float64 {
	actual := scores.ArgMax()
	correct := 0.0
	for i := range actual {
		if y[i][actual[i]] > 0 {
			correct += 1
		}
	}
	return correct / float64(len(y))
}

func (t *NeuralNet) divide(percent int, xIn [][]float64, yIn [][]byte) (*Matrix, [][]byte, *Matrix, [][]byte) {
	predictionLength := int(math.Floor(float64(len(xIn)) / float64(percent)))
	x := t.makeMatrix(xIn[:len(xIn)-predictionLength])
	y := yIn[:len(xIn)-predictionLength]
	xPred := t.makeMatrix(xIn[len(xIn)-predictionLength:])
	yPred := yIn[len(xIn)-predictionLength:]
	return x, y, xPred, yPred
}

func (t *NeuralNet) makeMatrix(inputX [][]float64) *Matrix {
	dataX := make([]float64, 0)
	for i := range inputX {
		dataX = append(dataX, inputX[i]...)
	}
	return NewMatrixF(dataX, len(inputX), len(inputX[0]))
}
