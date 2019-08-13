package gomlp

import (
	"math"

	mh "github.com/korn33v/gomlp/matrix"
	"gonum.org/v1/gonum/mat"
)

// Network ... s
type Network struct {
	Inputs        int
	Hiddens       int
	Outputs       int
	HiddenWeights *mat.Dense
	OutputWeights *mat.Dense
	LearningRate  float64
}

// CreateNetwork creates a neural network with random weights
func CreateNetwork(input, hidden, output int, rate float64) (net *Network) {
	net = &Network{
		Inputs:       input,
		Hiddens:      hidden,
		Outputs:      output,
		LearningRate: rate,
	}
	net.HiddenWeights = mat.NewDense(net.Hiddens, net.Inputs, mh.RandomArray(net.Inputs*net.Hiddens, float64(net.Inputs)))
	net.OutputWeights = mat.NewDense(net.Outputs, net.Hiddens, mh.RandomArray(net.Hiddens*net.Outputs, float64(net.Hiddens)))

	return net
}

// Train the neural network
func (net *Network) Train(inputData []float64, targetData []float64) {
	// feedforward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := mh.Dot(net.HiddenWeights, inputs)
	hiddenOutputs := mh.Apply(sigmoid, hiddenInputs)
	finalInputs := mh.Dot(net.OutputWeights, hiddenOutputs)
	finalOutputs := mh.Apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := mh.Subtract(targets, finalOutputs)
	hiddenErrors := mh.Dot(net.OutputWeights.T(), outputErrors)

	// backpropagate
	net.OutputWeights = mh.Add(net.OutputWeights,
		mh.Scale(net.LearningRate,
			mh.Dot(mh.Multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.HiddenWeights = mh.Add(net.HiddenWeights,
		mh.Scale(net.LearningRate,
			mh.Dot(mh.Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

// Predict uses the neural network to predict the value given input data
func (net *Network) Predict(inputData []float64) mat.Matrix {
	// feedforward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := mh.Dot(net.HiddenWeights, inputs)
	hiddenOutputs := mh.Apply(sigmoid, hiddenInputs)
	finalInputs := mh.Dot(net.OutputWeights, hiddenOutputs)
	finalOutputs := mh.Apply(sigmoid, finalInputs)

	return finalOutputs
}

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)

	return mh.Multiply(m, mh.Subtract(ones, m)) // m * (1 - m)
}
