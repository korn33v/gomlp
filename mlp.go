package gomlp

import (
	"math"

	mh "github.com/korn33v/gomlp/matrix"
	"gonum.org/v1/gonum/mat"
)

// MLP is a feedforward neural network with 3 layers
type MLP interface {
	Train(inputData []float64, targetData []float64)
	Predict(inputData []float64) mat.Matrix
}

// Network is an implementation of MLP
type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}

// New creates a neural network with random weights
func New(input, hidden, output int, rate float64) (net *Network) {
	net = &Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}
	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, mh.RandomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, mh.RandomArray(net.hiddens*net.outputs, float64(net.hiddens)))

	return net
}

// Train the neural network
func (net *Network) Train(inputData []float64, targetData []float64) {
	// feedforward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := mh.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := mh.Apply(sigmoid, hiddenInputs)
	finalInputs := mh.Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := mh.Apply(sigmoid, finalInputs)

	// find errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := mh.Subtract(targets, finalOutputs)
	hiddenErrors := mh.Dot(net.outputWeights.T(), outputErrors)

	// backpropagate
	net.outputWeights = mh.Add(net.outputWeights,
		mh.Scale(net.learningRate,
			mh.Dot(mh.Multiply(outputErrors, sigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = mh.Add(net.hiddenWeights,
		mh.Scale(net.learningRate,
			mh.Dot(mh.Multiply(hiddenErrors, sigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)
}

// Predict uses the neural network to predict the value given input data
func (net *Network) Predict(inputData []float64) mat.Matrix {
	// feedforward
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := mh.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := mh.Apply(sigmoid, hiddenInputs)
	finalInputs := mh.Dot(net.outputWeights, hiddenOutputs)
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
