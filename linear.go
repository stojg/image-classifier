package main

func linear(x *Vector, W *Matrix, b *Vector) *Matrix {
	return W.MulVec(x).Add(b.Transpose())
}
