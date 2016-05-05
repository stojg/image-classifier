package main

type Trainer interface {
	Train(trainingImages [][][]float64)
}

type Predictor interface {
	Predict(testData []float64) (testLabels []float64)
}
