package main

import (
	"bitbucket.org/binet/go-gnuplot/pkg/gnuplot"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

type NeuralNet struct {
	HiddenNeurons int
	Alpha         float64
	Lambda        float64

	sync.Mutex
	W1 *Matrix
	W2 *Matrix

	numBatches int
	numEpochs  int
	log        bool
	plot       bool
	costPlot   *gnuplot.Plotter
}

// @todo add more depth with convnets for image processing
// @todo use all CPU cores
// @todo link with a proper C lib for faster linear algebra (e.g. https://github.com/gonum/blas)
func (t *NeuralNet) Train(xTr, yTr, xCv, yCv [][]float64) (float64, float64) {

	if t.plot {
		t.initPlots()
		defer t.costPlot.Close()
	}

	inputNeurons := len(xTr[0])
	outputNeurons := len(yTr[0])

	// initialize parameters (weights) randomly for input -> hidden layer
	t.W1 = NewRandomMatrix(t.HiddenNeurons, inputNeurons+1).ScalarMul(0.12)

	// initialize parameters (weights) randomly for hidden-> output layer
	t.W2 = NewRandomMatrix(outputNeurons, t.HiddenNeurons+1).ScalarMul(0.12)

	// these are used so that we can update gnuplots with the data
	var (
		trainingCosts    []float64
		trainingEpochs   []float64
		validationCosts  []float64
		validationEpochs []float64
	)

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var wg sync.WaitGroup

	for epoch := 1; epoch < t.numEpochs+1; epoch++ {
		xBatches, yBatches := t.randomisedBatches(t.numBatches, xTr, yTr)
		dW1 := NewZeros(t.W1.Rows, t.W1.Cols)
		dW2 := NewZeros(t.W2.Rows, t.W2.Cols)

		aChan := make(chan *Matrix, len(xBatches))
		bChan := make(chan *Matrix, len(xBatches))

		for i := range xBatches {
			// calculate each batch in it's own go routine so we utilize as many CPU resources as possible
			wg.Add(1)
			go func(idx int) {
				_, a, b := t.costFunction(xBatches[idx], yBatches[idx], t.Lambda)
				aChan <- a
				bChan <- b
			}(i)
			// drain each batch in it's own go routine
			go func() {
				dW1 = dW1.Add(<-aChan)
				dW2 = dW2.Add(<-bChan)
				wg.Done()
			}()
		}
		wg.Wait()

		// parameter updates
		t.W2 = t.W2.Sub(dW2.ScalarMul(t.Alpha))
		t.W1 = t.W1.Sub(dW1.ScalarMul(t.Alpha))

		select {
		case <-ticker.C:

			jTrain, _, _ := t.costFunction(NewMatrix(xTr), NewMatrix(yTr), 0)
			trainingCosts = append(trainingCosts, jTrain)
			trainingEpochs = append(trainingEpochs, float64(epoch))

			jValidation, _, _ := t.costFunction(NewMatrix(xCv), NewMatrix(yCv), 0)
			validationCosts = append(validationCosts, jValidation)
			validationEpochs = append(validationEpochs, float64(epoch))

			if t.log {
				log.Printf("epoch %d:\t%f\t%f", epoch, trainingCosts[len(trainingCosts)-1], validationCosts[len(validationCosts)-1])
			}
			if t.plot {
				t.plotCost(trainingCosts, trainingEpochs, validationCosts, validationEpochs)
			}
		default:
		}
	}

	jTrain, _, _ := t.costFunction(NewMatrix(xTr), NewMatrix(yTr), 0)
	trainingCosts = append(trainingCosts, jTrain)

	// check the cost for the validation set
	jValidation, _, _ := t.costFunction(NewMatrix(xCv), NewMatrix(yCv), 0)
	validationCosts = append(validationCosts, jValidation)

	if len(validationCosts) != 0 && len(trainingCosts) != 0 && t.plot {
		t.plotCost(trainingCosts, trainingEpochs, validationCosts, validationEpochs)
	}
	return trainingCosts[len(trainingCosts)-1], validationCosts[len(validationCosts)-1]
}

func (t *NeuralNet) Predict(input []float64) []int {
	xTe := NewMatrixF(input, 1, len(input))
	a1 := xTe.AddBias()
	z2 := a1.Dot(t.W1.T())
	a2 := t.sigmoid(z2).AddBias()
	z3 := a2.Dot(t.W2.T())
	a3 := t.sigmoid(z3)
	return a3.ArgMax()
}

func (t *NeuralNet) Divide(xIn [][]float64, yIn [][]float64) (x, y, xPred, yPred [][]float64) {
	predictionLength := int(math.Floor(float64(len(xIn)) / 2))
	x = xIn[:len(xIn)-predictionLength]
	y = yIn[:len(xIn)-predictionLength]
	xPred = xIn[len(xIn)-predictionLength:]
	yPred = yIn[len(xIn)-predictionLength:]
	return x, y, xPred, yPred
}

func (t *NeuralNet) costFunction(x, y *Matrix, lambda float64) (J float64, gradW1 *Matrix, gradW2 *Matrix) {

	gradW1 = NewZeros(t.W1.Rows, t.W1.Cols)
	gradW2 = NewZeros(t.W2.Rows, t.W2.Cols)

	t.Lock()
	W1 := t.W1.Clone()
	W2 := t.W2.Clone()
	t.Unlock()

	// input
	a1 := x.AddBias()
	z2 := a1.Dot(W1.T())
	a2 := t.sigmoid(z2).AddBias()
	z3 := a2.Dot(W2.T())
	a3 := t.sigmoid(z3)

	J1 := y.ScalarMul(-1).ElementMul(a3.ElementLog())

	ones1 := NewOnes(y.Rows, y.Cols)
	ones2 := NewOnes(a3.Rows, a3.Cols)
	J2 := ones1.Sub(y).ElementMul(ones2.Sub(a3).ElementLog())

	Jreg1 := W1.RemoveBias().ElementSquare().Sum()
	Jreg2 := W2.RemoveBias().ElementSquare().Sum()

	m := float64(x.Rows)
	J = (J1.Sub(J2).Sum() / m) + (lambda * (Jreg1 + Jreg2) / (2 * m))

	d3 := a3.Sub(y)
	d2 := d3.Dot(W2.RemoveBias()).ElementMul(t.sigmoidPrime(z2))

	gradW1 = d2.T().Dot(a1).ScalarDiv(m)
	gradW2 = d3.T().Dot(a2).ScalarDiv(m)

	// add regularisation to gradients
	if lambda != 0 {
		gradW1Reg := W1.ZeroBias().ScalarDiv(lambda / m)
		gradW2Reg := W2.ZeroBias().ScalarDiv(lambda / m)
		gradW1.Add(gradW1Reg)
		gradW2.Add(gradW2Reg)
	}

	return J, gradW1, gradW2
}

func (t *NeuralNet) randomisedBatches(batchSize int, xAll [][]float64, yAll [][]float64) (X, Y []*Matrix) {
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

	for i := range xBatch {
		if len(xBatch[i]) == 0 {
			continue
		}
		X = append(X, NewMatrix(xBatch[i]))
	}

	for i := range yBatch {
		if len(yBatch[i]) == 0 {
			continue
		}
		Y = append(Y, NewMatrix(yBatch[i]))
	}

	return X, Y
}

func (t *NeuralNet) sigmoid(A *Matrix) *Matrix {
	res := A.Clone()
	for i := range A.Data {
		res.Data[i] = 1.0 / (1.0 + math.Exp(-res.Data[i]))
	}
	return res
}

func (t *NeuralNet) sigmoidPrime(A *Matrix) *Matrix {
	ones := NewOnes(A.Rows, A.Cols)
	return t.sigmoid(A).ElementMul(ones.Sub(t.sigmoid(A)))
}

// setup gnuplot for plotting the loss and accuracy (brew install gnuplot)
func (t *NeuralNet) initPlots() {
	var err error

	if t.costPlot, err = gnuplot.NewPlotter("", false, false); err != nil {
		panic(fmt.Sprintf("** err: %v\n", err))
	}

	t.costPlot.SetStyle("lines")
	title := fmt.Sprintf("set title \"Cost plot\"")
	t.costPlot.Cmd(title)

	t.costPlot.Cmd(fmt.Sprintf("set label 1 \"hidden neurons: %d\\nalpha: %f\\nlambda: %f\"", t.HiddenNeurons, t.Alpha, t.Lambda))
	t.costPlot.Cmd("set label 1 at graph 0.1, 0.95 tc default")
	t.costPlot.SetXLabel("epoch")
	t.costPlot.SetYLabel("cost")
	t.costPlot.Cmd("set yrange [0:]")
}

func (t *NeuralNet) plotCost(trainingCosts, trainingEpochs, validationCosts, validationEpochs []float64) {
	t.costPlot.ResetPlot()
	t.costPlot.PlotXY(trainingEpochs, trainingCosts, "training")
	t.costPlot.PlotXY(validationEpochs, validationCosts, "validation")
}
