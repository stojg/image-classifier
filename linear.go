package main

import (
	"math"
)

//  SVMLoss uses a (Multiclass) Support Vector Machine to calculate a "hinge loss"
// - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
//     with an appended bias dimension in the 3073-rd position (i.e. bias trick)
// - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
// - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
func SVMLoss(x *Vector, y int, W *Matrix) float64 {
	const delta = 1.0

	// this is a standard score calculation
	scores := W.MulVec(x)

	correctClassScore := scores.At(y, 0)

	loss := 0.0
	for i := 0; i < scores.rows; i++ {
		if i == y {
			continue
		}
		loss += math.Max(0, scores.At(i, 0)-correctClassScore+delta)
	}
	return loss
}

//  SoftMaxLoss calculates the "cross-entropy"
// - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
//     with an appended bias dimension in the 3073-rd position (i.e. bias trick)
// - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
// - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
func SoftMaxLoss(x *Vector, y int, W *Matrix) float64 {
	// this is a standard score calculation
	f := W.MulVec(x)

	// due to math instability on high values and the use of exp() and log()
	// shift the values of scores so that the highest number is 0
	f = f.ScalarSub(f.Max())

	// calculate the sum so we can normalise the value
	sum := f.ScalarExp().Sum()

	// val should now be in the range of [0.0..1.0]
	val := math.Exp(f.At(y, 0)) / sum

	// this is the cross-entropy loss calculation
	return -math.Log(val)

	// below is the full code to calculate all all classes loss,
	// code has been commented because of slight performance since it
	// calculates exp() and scalarDiv on all elements, even the ones we don't want
	//f = f.ScalarSub(f.Max())
	//sum := f.ScalarExp().Sum()
	//p := f.ScalarExp().ScalarDiv(sum)
	//return - math.Log(p.At(y,0))
}
