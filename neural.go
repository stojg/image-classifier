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
	labels   [][]float64
	W1       *Matrix
	bias1    *Matrix
	W2       *Matrix
	bias2    *Matrix
	log      bool
	plot     bool
	lossPlot *gnuplot.Plotter
	accPlot  *gnuplot.Plotter
}

// @todo print time per epoch
// @todo mini batch updates
// @todo add more depth with convnets for image processing
// @todo use all CPU cores
// @todo link with a proper C lib for faster linear algebra (e.g. https://github.com/gonum/blas)
func (t *NeuralNet) Train(xIn [][]float64, yIn [][]byte, numEpochs int) {

	const (
		reg      = 1e-3
		stepSize = 0.001
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
	hiddenNeurons := 26
	outputNeurons := len(y[0])

	// Put up a checkpoint so we can check that the data and loss is set up correctly
	if t.log {
		log.Printf("checkpoint: initial loss should be roughly %f (1 out of %d)", -math.Log(1/float64(outputNeurons)), outputNeurons)
	}

	// initialize parameters (weights) randomly for input -> hidden layer
	W1 := NewRandomMatrix(inputNeurons, hiddenNeurons).ScalarMul(0.01)
	bias1 := NewRandomMatrix(1, hiddenNeurons).ScalarMul(0.01)

	// initialize parameters (weights) randomly for hidden-> output layer
	W2 := NewRandomMatrix(hiddenNeurons, outputNeurons).ScalarMul(0.01)
	bias2 := NewRandomMatrix(1, outputNeurons).ScalarMul(0.01)

	// keep a running tally of the score between hidden -> output so that we can print
	// it after the epoch loop
	var outputLayer *Matrix

	// these are used so that we can update gnuplots with the data
	var (
		losses      []float64
		accuracies  []float64
		pAccuracies []float64
	)

	// The training loop
	for epoch := 0; epoch < numEpochs; epoch++ {

		// evaluate input -> hidden
		hiddenLayer := t.ReLu(x.Dot(W1).RowAdd(bias1))
		// final output layer "scores", note that there we don't need a ReLU activation
		outputLayer = hiddenLayer.Dot(W2).RowAdd(bias2)

		// loss
		loss := t.softMaxLoss(outputLayer, W1, y, reg)
		acc := t.accuracy(outputLayer, y) * 100

		// calculate how well we can predict
		pred := t.ReLu(xPred.Dot(W1).RowAdd(bias1))
		predAcc := t.accuracy(pred.Dot(W2).RowAdd(bias2), yPred) * 100

		if t.plot {
			losses = append(losses, loss)
			accuracies = append(accuracies, acc)
			pAccuracies = append(pAccuracies, predAcc)
		}

		if t.log && epoch%100 == 0 {
			log.Printf("epoch %d:  loss: %f,  accuracy %.1f%% / %.1f%%", epoch, loss, acc, predAcc)
			if t.plot {
				t.plotProgress(losses, accuracies, pAccuracies)
			}
		}

		// Start the backward propagation
		grad := t.softMaxGradient(outputLayer, y)

		dW2 := hiddenLayer.T().Dot(grad)
		db2 := grad.ColSum()

		dHidden := grad.Dot(W2.T())
		t.ReLuBackProp(dHidden, hiddenLayer)

		dW1 := x.T().Dot(dHidden)
		db1 := dHidden.ColSum()

		// add regularisation
		dW2 = dW2.Add(W2.ScalarMul(reg))
		dW1 =  dW1.Add(W1.ScalarMul(reg))

		// parameter update
		W2 = W2.Sub(dW2.ScalarMul(stepSize))
		bias2 = bias2.Sub(db2.ScalarMul(stepSize))

		// parameter update
		W1 = W1.Sub(dW1.ScalarMul(stepSize))
		bias1 = bias1.Sub(db1.ScalarMul(stepSize))
	}

	t.plotProgress(losses, accuracies, pAccuracies)

	// "close" the gnuplots
	t.lossPlot.Cmd("q")
	t.accPlot.Cmd("q")

	// save the final values so that Predict() can find them
	t.W2 = W2
	t.bias2 = bias2
	t.W1 = W1
	t.bias1 = bias1

	if t.log {
		loss := t.softMaxLoss(outputLayer, W1, y, reg)
		log.Printf("result: loss: %f, accuracy %.1f%%", loss, t.accuracy(outputLayer, y)*100)
	}
}


func (t *NeuralNet) Predict(input []float64) []int {
	// convert into a matrix
	xTe := NewMatrixF(input, 1, len(input))
	// evaluate class scores with a 2-layer Neural Network
	hiddenLayer := xTe.Dot(t.W1).RowAdd(t.bias1).ElementMax(0) // ReLU activation
	scores := hiddenLayer.Dot(t.W2).RowAdd(t.bias2)
	return scores.ArgMax()
}


func (t *NeuralNet) plotProgress(losses, acc, pacc []float64) {
	t.lossPlot.ResetPlot()
	t.lossPlot.PlotX(losses, "loss (cross-entropy)")
	t.accPlot.ResetPlot()
	t.accPlot.PlotX(acc, "test")
	t.accPlot.PlotX(pacc, "validation")

}

// calculate the loss function with an average cross-entropy
// cross-entropy gives us a way to express how different two probability
// distributions are, see http://colah.github.io/posts/2015-09-Visual-Information/
func (t *NeuralNet) softMaxLoss(a2, theta1 *Matrix, y [][]byte, reg float64) float64 {
	// get un normalized probabilities
	a2Exp := a2.Clone().ScalarExp()
	// normalize them for each example
	probs := a2Exp.ColDiv(a2Exp.RowSum())
	// this is the loss function
	correctLogProbs := probs.RowFinder(y).ScalarMinusLog()
	// and here is the softMaxLoss
	dataLoss := correctLogProbs.Sum() / float64(len(y))
	// calculate the regularization loss
	regLoss := 0.5 * reg * theta1.ElementMul(theta1).Sum()
	return dataLoss + regLoss
}


func (t *NeuralNet) softMaxGradient(A *Matrix, y [][]byte) *Matrix {
	// find the gradient descent for output
	dScores := A.Clone()
	for row := range y {
		for col := range y[row] {
			if y[row][col] > 0 {
				// by decreasing the hScore[j] that was the correct answer with
				// 1 we find the gradient descent in the direction where the
				// loss function is decreasing the most. But honestly, it's a bit
				// fuzzy for me how this actually works
				dScores.data[row*dScores.cols+col] -= 1
				break
			}
		}
	}
	// calculate the the average
	return dScores.ScalarDiv(float64(A.rows))
}

//  Rectified Linear unit
func (t *NeuralNet) ReLu(A *Matrix) *Matrix {
	return A.ElementMax(0)
}

func (t *NeuralNet) ReLuBackProp(dTheta, layer *Matrix) {
	// ReLu will not propagate back  any corrections
	// to a node that was correct (below 0)
	for i := range dTheta.data {
		if layer.data[i] < 0 {
			dTheta.data[i] = 0
		}
	}
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
