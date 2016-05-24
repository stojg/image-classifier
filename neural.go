package main

import (
	"bitbucket.org/binet/go-gnuplot/pkg/gnuplot"
	"fmt"
	"log"
	"math"
	"math/rand"
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
	stepSize float64
	reg      float64
}

// @todo print time per epoch
// @todo add more depth with convnets for image processing
// @todo use all CPU cores
// @todo link with a proper C lib for faster linear algebra (e.g. https://github.com/gonum/blas)
func (t *NeuralNet) Train(xIn [][]float64, yIn [][]byte, numEpochs, numBatches int) {

	t.reg = 1e-3
	t.stepSize = 0.001

	// setup gnuplots
	t.initPlots()
	defer t.lossPlot.Close()
	defer t.accPlot.Close()

	inputNeurons := len(xIn[0])
	hiddenNeurons := 36
	outputNeurons := len(yIn[0])

	// Put up a checkpoint so we can check that the data and loss is set up correctly
	if t.log {
		log.Printf("checkpoint: initial loss should be roughly %f (1 out of %d)", -math.Log(1/float64(outputNeurons)), outputNeurons)
	}

	// initialize parameters (weights) randomly for input -> hidden layer
	t.W1 = NewRandomMatrix(inputNeurons, hiddenNeurons).ScalarMul(0.01)
	t.bias1 = NewRandomMatrix(1, hiddenNeurons).ScalarMul(0.01)

	// initialize parameters (weights) randomly for hidden-> output layer
	t.W2 = NewRandomMatrix(hiddenNeurons, outputNeurons).ScalarMul(0.01)
	t.bias2 = NewRandomMatrix(1, outputNeurons).ScalarMul(0.01)

	// keep a running tally of the score between hidden -> output so that we can print
	// it after the epoch loop
	var outputLayer *Matrix

	// these are used so that we can update gnuplots with the data
	var (
		losses      []float64
		accuracies  []float64
		pAccuracies []float64
	)

	xAll, yAll, xPred, yPred := t.divide(10, xIn, yIn)

	// The training loop
	for epoch := 0; epoch < numEpochs; epoch++ {

		xBatches, yBatches := t.getBatches(numBatches, xAll, yAll)

		dW1 := NewZerosMatrix(t.W1.rows, t.W1.cols)
		dBias1 := NewZerosMatrix(t.bias1.rows, t.bias1.cols)
		dW2 := NewZerosMatrix(t.W2.rows, t.W2.cols)
		dBias2 := NewZerosMatrix(t.bias2.rows, t.bias2.cols)

		for i := range xBatches {
			a, b, c, d := t.GradientDescent(xBatches[i], yBatches[i])
			dW1 = dW1.Add(a)
			dBias1 = dBias1.Add(b)
			dW2 = dW2.Add(c)
			dBias2 = dBias2.Add(d)
		}

		// parameter update
		t.W2 = t.W2.Sub(dW2.ScalarMul(t.stepSize))
		t.bias2 = t.bias2.Sub(dBias2.ScalarMul(t.stepSize))

		// parameter update
		t.W1 = t.W1.Sub(dW1.ScalarMul(t.stepSize))
		t.bias1 = t.bias1.Sub(dBias1.ScalarMul(t.stepSize))

		// calculate loss
		hiddenLayer := t.ReLu(NewMatrix(xAll).Dot(t.W1).RowAdd(t.bias1))
		outputLayer = hiddenLayer.Dot(t.W2).RowAdd(t.bias2)
		loss := t.softMaxLoss(outputLayer, t.W1, yAll, t.reg)
		trainingError := t.error(outputLayer, yAll) * 100

		// calculate how well we can predict
		pHiddenLayer := t.ReLu(NewMatrix(xPred).Dot(t.W1).RowAdd(t.bias1))
		pOutputLayer := pHiddenLayer.Dot(t.W2).RowAdd(t.bias2)
		predictionError := t.error(pOutputLayer, yPred) * 100

		if t.plot {
			losses = append(losses, loss)
			accuracies = append(accuracies, trainingError)
			pAccuracies = append(pAccuracies, predictionError)
		}

		if t.log && epoch%500 == 0 {
			log.Printf("epoch %d:  loss: %f,  error %.1f%%/%.1f%%", epoch, loss, trainingError, predictionError)
			if t.plot {
				t.plotProgress(losses, accuracies, pAccuracies)
			}
		}
	}
	t.plotProgress(losses, accuracies, pAccuracies)
}

func (t *NeuralNet) Predict(input []float64) []int {
	// convert into a matrix
	xTe := NewMatrixF(input, 1, len(input))
	// evaluate class scores with a 2-layer Neural Network
	hiddenLayer := t.ReLu(xTe.Dot(t.W1).RowAdd(t.bias1))
	scores := hiddenLayer.Dot(t.W2).RowAdd(t.bias2)
	return scores.ArgMax()
}

func (t *NeuralNet) getBatches(batchSize int, xAll [][]float64, yAll [][]byte) ([]*Matrix, [][][]byte) {
	for i := range xAll {
		j := rand.Intn(i + 1)
		xAll[i], xAll[j] = xAll[j], xAll[i]
		yAll[i], yAll[j] = yAll[j], yAll[i]
	}

	xBatch := make([][][]float64, batchSize)
	yBatch := make([][][]byte, batchSize)

	for i := range xAll {
		bIdx := rand.Intn(batchSize)
		xBatch[bIdx] = append(xBatch[bIdx], xAll[i])
		yBatch[bIdx] = append(yBatch[bIdx], yAll[i])
	}

	var X []*Matrix
	for i := range xBatch {
		if len(xBatch[i]) == 0 {
			continue
		}
		X = append(X, NewMatrix(xBatch[i]))
	}

	return X, yBatch
}

func (t *NeuralNet) GradientDescent(x *Matrix, y [][]byte) (*Matrix, *Matrix, *Matrix, *Matrix) {

	// evaluate input -> hidden
	hiddenLayer := t.ReLu(x.Dot(t.W1).RowAdd(t.bias1))
	// final output layer "scores", note that there we don't need a ReLU activation
	outputLayer := hiddenLayer.Dot(t.W2).RowAdd(t.bias2)

	// Start the backward propagation
	grad := t.softMaxGradient(outputLayer, y)

	dW2 := hiddenLayer.T().Dot(grad)
	db2 := grad.ColSum()

	dHidden := grad.Dot(t.W2.T())
	t.ReLuBackProp(dHidden, hiddenLayer)

	dW1 := x.T().Dot(dHidden)
	db1 := dHidden.ColSum()

	// add regularisation
	dW2 = dW2.Add(t.W2.ScalarMul(t.reg))
	dW1 = dW1.Add(t.W1.ScalarMul(t.reg))

	return dW1, db1, dW2, db2
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
	accPlot.SetYLabel("error %")
	accPlot.Cmd("set yrange [0:110]")

	t.lossPlot = lossPlot
	t.accPlot = accPlot
}

// accuracy returns a accuracy between 0.0 - 1.0 on how correct a score (classification)
// matrix is compared to the actual (Y) labels
func (t *NeuralNet) error(scores *Matrix, y [][]byte) float64 {
	actual := scores.ArgMax()
	correct := 0.0
	for i := range actual {
		if y[i][actual[i]] > 0 {
			correct += 1
		}
	}
	return 1 - (correct / float64(len(y)))
}

func (t *NeuralNet) divide(percent int, xIn [][]float64, yIn [][]byte) ([][]float64, [][]byte, [][]float64, [][]byte) {
	predictionLength := int(math.Floor(float64(len(xIn)) / float64(percent)))
	x := xIn[:len(xIn)-predictionLength]
	y := yIn[:len(xIn)-predictionLength]
	xPred := xIn[len(xIn)-predictionLength:]
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
