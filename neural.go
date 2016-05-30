package main

import (
	"bitbucket.org/binet/go-gnuplot/pkg/gnuplot"
	"fmt"
	"math"
	"math/rand"
)

type NeuralNet struct {
	labels   [][]float64
	W1       *Matrix
	W2       *Matrix
	log      bool
	plot     bool
	lossPlot *gnuplot.Plotter
	accPlot  *gnuplot.Plotter
	alpha    float64
	lambda   float64
}

// @todo print time per epoch
// @todo add more depth with convnets for image processing
// @todo use all CPU cores
// @todo link with a proper C lib for faster linear algebra (e.g. https://github.com/gonum/blas)
func (t *NeuralNet) Train(xIn [][]float64, yIn [][]float64, hiddenNeurons, numEpochs, numBatches int) (float64, float64) {

	t.initPlots()
	defer t.lossPlot.Close()
	defer t.accPlot.Close()
	predictions := 50

	inputNeurons := len(xIn[0])
	outputNeurons := len(yIn[0])

	// initialize parameters (weights) randomly for input -> hidden layer
	t.W1 = NewRandomMatrix(hiddenNeurons, inputNeurons+1).ScalarMul(0.12)

	// initialize parameters (weights) randomly for hidden-> output layer
	t.W2 = NewRandomMatrix(outputNeurons, hiddenNeurons+1).ScalarMul(0.12)

	// these are used so that we can update gnuplots with the data
	var (
		losses      []float64
		accuracies  []float64
		pAccuracies []float64
	)

	xTr, yTr, xCv, yCv := t.divide(predictions, xIn, yIn)

	// The training loop
	for epoch := 0; epoch < numEpochs; epoch++ {
		xBatches, yBatches := t.randomisedBatches(numBatches, xTr, yTr)
		dW1 := NewZeros(t.W1.Rows, t.W1.Cols)
		dW2 := NewZeros(t.W2.Rows, t.W2.Cols)
		var J float64
		for i := range xBatches {
			Jtemp, a, b := t.miniBatch(xBatches[i], yBatches[i])
			J += Jtemp
			dW1 = dW1.Add(a)
			dW2 = dW2.Add(b)
		}
		losses = append(losses, (J / float64(len(xBatches))))
		// parameter updates
		t.W2 = t.W2.Sub(dW2.ScalarMul(t.alpha))
		t.W1 = t.W1.Sub(dW1.ScalarMul(t.alpha))
	}
	t.plotProgress(losses, accuracies, pAccuracies)

	trError := t.CalcError(xTr, yTr)
	cvError := t.CalcError(xCv, yCv)
	return trError, cvError

}

func (t *NeuralNet) Predict(input []float64) []int {

	// convert into a matrix
	xTe := NewMatrixF(input, 1, len(input))
	a1 := xTe.AddBias()
	z2 := a1.Dot(t.W1.T())
	a2 := t.Sigmoid(z2).AddBias()
	z3 := a2.Dot(t.W2.T())
	a3 := t.Sigmoid(z3)

	return a3.ArgMax()
}

func (t *NeuralNet) randomisedBatches(batchSize int, xAll [][]float64, yAll [][]float64) ([]*Matrix, []*Matrix) {
	for i := range xAll {
		j := rand.Intn(i + 1)
		xAll[i], xAll[j] = xAll[j], xAll[i]
		yAll[i], yAll[j] = yAll[j], yAll[i]
	}

	xBatch := make([][][]float64, batchSize)
	yBatch := make([][][]float64, batchSize)

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

	var Y []*Matrix
	for i := range yBatch {
		if len(yBatch[i]) == 0 {
			continue
		}
		Y = append(Y, NewMatrix(yBatch[i]))
	}

	return X, Y
}

func (t *NeuralNet) Sigmoid(A *Matrix) *Matrix {
	res := A.Clone()
	for i := range A.Data {
		res.Data[i] = 1.0 / (1.0 + math.Exp(-res.Data[i]))
	}
	return res
}

func (t *NeuralNet) SigmoidPrime(A *Matrix) *Matrix {
	ones := NewOnes(A.Rows, A.Cols)
	return t.Sigmoid(A).ElementMul(ones.Sub(t.Sigmoid(A)))
}

func (t *NeuralNet) miniBatch(x, y *Matrix) (float64, *Matrix, *Matrix) {

	gradW1 := NewZeros(t.W1.Rows, t.W1.Cols)
	gradW2 := NewZeros(t.W2.Rows, t.W2.Cols)

	// input
	a1 := x.AddBias()
	z2 := a1.Dot(t.W1.T())
	a2 := t.Sigmoid(z2).AddBias()
	z3 := a2.Dot(t.W2.T())
	a3 := t.Sigmoid(z3)

	J1 := y.ScalarMul(-1).ElementMul(a3.ElementLog())

	ones1 := NewOnes(y.Rows, y.Cols)
	ones2 := NewOnes(a3.Rows, a3.Cols)
	J2 := ones1.Sub(y).ElementMul(ones2.Sub(a3).ElementLog())

	Jreg1 := t.W1.RemoveBias().ElementSquare().Sum()
	Jreg2 := t.W2.RemoveBias().ElementSquare().Sum()

	m := float64(x.Rows)
	J := (J1.Sub(J2).Sum() / m) + (t.lambda * (Jreg1 + Jreg2) / (2 * m))

	d3 := a3.Sub(y)
	d2 := d3.Dot(t.W2.RemoveBias()).ElementMul(t.SigmoidPrime(z2))

	gradW1 = d2.T().Dot(a1).ScalarDiv(m)
	gradW2 = d3.T().Dot(a2).ScalarDiv(m)

	// add regularisation to gradients

	gradW1Reg := t.W1.ZeroBias().ScalarDiv(t.lambda / m)
	gradW2Reg := t.W2.ZeroBias().ScalarDiv(t.lambda / m)

	gradW1.Add(gradW1Reg)
	gradW2.Add(gradW2Reg)
	return J, gradW1, gradW2
}

func (t *NeuralNet) Forward(x *Matrix) *Matrix {
	// input
	a1 := x.AddBias()
	z2 := a1.Dot(t.W1.T())
	a2 := t.Sigmoid(z2).AddBias()
	z3 := a2.Dot(t.W2.T())
	a3 := t.Sigmoid(z3)
	return a3
}

func (t *NeuralNet) CalcError(xInput, yInput [][]float64) float64 {
	x := NewMatrix(xInput)
	y := NewMatrix(yInput)
	a3 := t.Forward(x)
	return a3.Sub(y).ElementSquare().Sum() / (2 * float64(len(xInput)))
}

func (t *NeuralNet) plotProgress(losses, acc, pacc []float64) {
	t.lossPlot.ResetPlot()
	t.lossPlot.PlotX(losses, "loss (cross-entropy)")
	t.accPlot.ResetPlot()
	t.accPlot.PlotX(acc, "test")
	t.accPlot.PlotX(pacc, "validation")
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

func (t *NeuralNet) divide(percent int, xIn [][]float64, yIn [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	predictionLength := int(math.Floor(float64(len(xIn)) / float64(100.0/percent)))
	x := xIn[:len(xIn)-predictionLength]
	y := yIn[:len(xIn)-predictionLength]
	xPred := xIn[len(xIn)-predictionLength:]
	yPred := yIn[len(xIn)-predictionLength:]

	return x, y, xPred, yPred
}
