package org.omnaest.utils.neuralnetwork.internal;

import java.util.Random;

import org.omnaest.utils.neuralnetwork.domain.ActivationFunction;

/**
 * One fully connected layer: a weight matrix, a bias vector and an {@link ActivationFunction}.
 * <p>
 * The layer caches the input and the activation of its last {@link #forward(double[])} call, because {@link #backward(double[], double)} needs both and
 * recomputing them would double the work. That cache is exactly why a layer must not be shared between threads.
 */
public class Layer
{
    /**
     * <code>weights[neuronIndex][inputIndex]</code>
     */
    private final double[][] weights;

    private final double[] biases;

    private final ActivationFunction activationFunction;

    private double[] lastInput;

    private double[] lastActivation;

    /**
     * Initializes the weights by Xavier uniform sampling, i.e. within <code>&plusmn;sqrt(6/(inputSize+neuronCount))</code>. Scaling the range by the layer
     * size keeps the signal variance roughly constant across layers, which is what stops a deeper network from saturating before it ever learns anything.
     * Biases start at zero.
     */
    public Layer(int inputSize, int neuronCount, ActivationFunction activationFunction, Random random)
    {
        this.activationFunction = activationFunction;
        this.weights = new double[neuronCount][inputSize];
        this.biases = new double[neuronCount];

        double weightLimit = Math.sqrt(6.0 / (inputSize + neuronCount));
        for (int neuronIndex = 0; neuronIndex < neuronCount; neuronIndex++)
        {
            for (int inputIndex = 0; inputIndex < inputSize; inputIndex++)
            {
                this.weights[neuronIndex][inputIndex] = (random.nextDouble() * 2.0 - 1.0) * weightLimit;
            }
        }
    }

    public double[] forward(double[] input)
    {
        if (input.length != this.getInputSize())
        {
            throw new IllegalArgumentException("Layer expects an input of size " + this.getInputSize() + " but got " + input.length);
        }

        double[] activation = new double[this.getNeuronCount()];
        for (int neuronIndex = 0; neuronIndex < activation.length; neuronIndex++)
        {
            double weightedInput = this.biases[neuronIndex];
            double[] neuronWeights = this.weights[neuronIndex];
            for (int inputIndex = 0; inputIndex < input.length; inputIndex++)
            {
                weightedInput += neuronWeights[inputIndex] * input[inputIndex];
            }
            activation[neuronIndex] = this.activationFunction.activate(weightedInput);
        }

        this.lastInput = input;
        this.lastActivation = activation;
        return activation;
    }

    /**
     * Applies one gradient descent step to this layer's weights and biases and hands the loss gradient on to the layer before it.
     *
     * @param lossGradientByActivation
     *            partial derivative of the loss with respect to each of this layer's outputs
     * @param learningRate
     *            step size; pass <code>0</code> to compute the incoming gradient without changing any weight
     * @return partial derivative of the loss with respect to each of this layer's <i>inputs</i>, which is the previous layer's
     *         <code>lossGradientByActivation</code>
     */
    public double[] backward(double[] lossGradientByActivation, double learningRate)
    {
        if (this.lastActivation == null)
        {
            throw new IllegalStateException("backward(..) requires a preceding forward(..) call on the same layer");
        }

        double[] lossGradientByInput = new double[this.getInputSize()];
        for (int neuronIndex = 0; neuronIndex < this.getNeuronCount(); neuronIndex++)
        {
            double neuronDelta = lossGradientByActivation[neuronIndex]
                    * this.activationFunction.derivativeOfActivated(this.lastActivation[neuronIndex]);
            double[] neuronWeights = this.weights[neuronIndex];

            for (int inputIndex = 0; inputIndex < neuronWeights.length; inputIndex++)
            {
                // the incoming gradient must be accumulated from the weight as it was during the forward pass, so read before updating
                lossGradientByInput[inputIndex] += neuronWeights[inputIndex] * neuronDelta;
                neuronWeights[inputIndex] -= learningRate * neuronDelta * this.lastInput[inputIndex];
            }
            this.biases[neuronIndex] -= learningRate * neuronDelta;
        }
        return lossGradientByInput;
    }

    public int getInputSize()
    {
        return this.weights.length == 0 ? 0 : this.weights[0].length;
    }

    public int getNeuronCount()
    {
        return this.weights.length;
    }

    /**
     * The live weight matrix, not a copy. Exposed for gradient verification within this internal package.
     */
    public double[][] getWeights()
    {
        return this.weights;
    }

    /**
     * The live bias vector, not a copy. Exposed for gradient verification within this internal package.
     */
    public double[] getBiases()
    {
        return this.biases;
    }

    public ActivationFunction getActivationFunction()
    {
        return this.activationFunction;
    }
}
