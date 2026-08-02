package org.omnaest.utils.neuralnetwork.internal;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.omnaest.utils.neuralnetwork.NeuralNetworkUtils;
import org.omnaest.utils.neuralnetwork.domain.DefaultActivationFunction;
import org.omnaest.utils.neuralnetwork.domain.DefaultLossFunction;
import org.omnaest.utils.neuralnetwork.domain.LossFunction;
import org.omnaest.utils.neuralnetwork.domain.TrainingExample;
import org.omnaest.utils.neuralnetwork.domain.TrainingSet;

/**
 * Verifies backpropagation against numerical differentiation.
 * <p>
 * A convergence test only shows that training ends up somewhere reasonable; a wrong sign or a misplaced index in the backward pass can still leave a network
 * that learns, just worse. Comparing every analytic partial derivative against a central difference of the loss checks the derivation itself, which is where
 * the realistic bugs of a hand written backpropagation live.
 */
public class BackpropagationGradientTest
{
    private static final double PERTURBATION = 1e-6;

    private static final double TOLERANCE = 1e-8;

    /**
     * With a learning rate of exactly one, a single update step subtracts the gradient itself, so <code>weightBefore - weightAfter</code> <i>is</i> the
     * analytic partial derivative. That lets the real training code path be measured without exposing a gradient accessor nobody else would need.
     */
    private static final double LEARNING_RATE_YIELDING_THE_RAW_GRADIENT = 1.0;

    @Test
    public void testAnalyticWeightGradientMatchesNumericalGradient() throws Exception
    {
        NeuralNetworkImpl network = createNetwork();
        TrainingExample example = createExample();
        LossFunction lossFunction = DefaultLossFunction.MEAN_SQUARED_ERROR;

        double[][][] numericalGradient = calculateNumericalWeightGradient(network, example, lossFunction);
        double[][][] weightsBefore = copyWeights(network);
        network.applyBackpropagation(example, lossFunction, LEARNING_RATE_YIELDING_THE_RAW_GRADIENT);

        for (int layerIndex = 0; layerIndex < network.getLayers()
                                                     .size(); layerIndex++)
        {
            double[][] weightsAfter = network.getLayers()
                                             .get(layerIndex)
                                             .getWeights();
            for (int neuronIndex = 0; neuronIndex < weightsAfter.length; neuronIndex++)
            {
                for (int inputIndex = 0; inputIndex < weightsAfter[neuronIndex].length; inputIndex++)
                {
                    double analyticGradient = weightsBefore[layerIndex][neuronIndex][inputIndex] - weightsAfter[neuronIndex][inputIndex];
                    assertEquals("Weight gradient mismatch at layer " + layerIndex + ", neuron " + neuronIndex + ", input " + inputIndex,
                                 numericalGradient[layerIndex][neuronIndex][inputIndex], analyticGradient, TOLERANCE);
                }
            }
        }
    }

    @Test
    public void testAnalyticBiasGradientMatchesNumericalGradient() throws Exception
    {
        NeuralNetworkImpl network = createNetwork();
        TrainingExample example = createExample();
        LossFunction lossFunction = DefaultLossFunction.MEAN_SQUARED_ERROR;

        double[][] numericalGradient = calculateNumericalBiasGradient(network, example, lossFunction);
        double[][] biasesBefore = copyBiases(network);
        network.applyBackpropagation(example, lossFunction, LEARNING_RATE_YIELDING_THE_RAW_GRADIENT);

        for (int layerIndex = 0; layerIndex < network.getLayers()
                                                     .size(); layerIndex++)
        {
            double[] biasesAfter = network.getLayers()
                                          .get(layerIndex)
                                          .getBiases();
            for (int neuronIndex = 0; neuronIndex < biasesAfter.length; neuronIndex++)
            {
                double analyticGradient = biasesBefore[layerIndex][neuronIndex] - biasesAfter[neuronIndex];
                assertEquals("Bias gradient mismatch at layer " + layerIndex + ", neuron " + neuronIndex,
                             numericalGradient[layerIndex][neuronIndex], analyticGradient, TOLERANCE);
            }
        }
    }

    /**
     * Deliberately asymmetric: three inputs, two hidden layers of different widths and mixed activations, two outputs. A square topology would hide an index
     * swap between the neuron and the input dimension.
     */
    private static NeuralNetworkImpl createNetwork()
    {
        return (NeuralNetworkImpl) NeuralNetworkUtils.newNetwork()
                                                     .withInputSize(3)
                                                     .addLayer(4, DefaultActivationFunction.TANH)
                                                     .addLayer(5, DefaultActivationFunction.RELU)
                                                     .addLayer(2, DefaultActivationFunction.SIGMOID)
                                                     .withRandomSeed(42)
                                                     .build();
    }

    private static TrainingExample createExample()
    {
        return TrainingExample.of(new double[] { 0.3, -0.7, 0.5 }, new double[] { 1.0, 0.0 });
    }

    private static double[][][] calculateNumericalWeightGradient(NeuralNetworkImpl network, TrainingExample example, LossFunction lossFunction)
    {
        double[][][] gradient = new double[network.getLayers()
                                                  .size()][][];
        for (int layerIndex = 0; layerIndex < gradient.length; layerIndex++)
        {
            double[][] weights = network.getLayers()
                                        .get(layerIndex)
                                        .getWeights();
            gradient[layerIndex] = new double[weights.length][];
            for (int neuronIndex = 0; neuronIndex < weights.length; neuronIndex++)
            {
                gradient[layerIndex][neuronIndex] = new double[weights[neuronIndex].length];
                for (int inputIndex = 0; inputIndex < weights[neuronIndex].length; inputIndex++)
                {
                    int capturedNeuronIndex = neuronIndex;
                    int capturedInputIndex = inputIndex;
                    gradient[layerIndex][neuronIndex][inputIndex] = calculateCentralDifference(network, example, lossFunction, weights[neuronIndex][inputIndex],
                                                                                              perturbedValue -> weights[capturedNeuronIndex][capturedInputIndex] = perturbedValue);
                }
            }
        }
        return gradient;
    }

    private static double[][] calculateNumericalBiasGradient(NeuralNetworkImpl network, TrainingExample example, LossFunction lossFunction)
    {
        double[][] gradient = new double[network.getLayers()
                                                .size()][];
        for (int layerIndex = 0; layerIndex < gradient.length; layerIndex++)
        {
            double[] biases = network.getLayers()
                                     .get(layerIndex)
                                     .getBiases();
            gradient[layerIndex] = new double[biases.length];
            for (int neuronIndex = 0; neuronIndex < biases.length; neuronIndex++)
            {
                int capturedNeuronIndex = neuronIndex;
                gradient[layerIndex][neuronIndex] = calculateCentralDifference(network, example, lossFunction, biases[neuronIndex],
                                                                               perturbedValue -> biases[capturedNeuronIndex] = perturbedValue);
            }
        }
        return gradient;
    }

    /**
     * <code>(loss(parameter + h) - loss(parameter - h)) / 2h</code>, restoring the original parameter afterwards. The central form is used rather than the
     * one sided one because its error falls off with h&sup2;, which is what makes a 1e-8 tolerance achievable at all.
     */
    private static double calculateCentralDifference(NeuralNetworkImpl network, TrainingExample example, LossFunction lossFunction, double parameterValue,
                                                     ParameterWriter parameterWriter)
    {
        TrainingSet trainingSet = TrainingSet.of(example);

        parameterWriter.write(parameterValue + PERTURBATION);
        double lossAbove = network.calculateLoss(trainingSet, lossFunction);

        parameterWriter.write(parameterValue - PERTURBATION);
        double lossBelow = network.calculateLoss(trainingSet, lossFunction);

        parameterWriter.write(parameterValue);
        return (lossAbove - lossBelow) / (2.0 * PERTURBATION);
    }

    private static double[][][] copyWeights(NeuralNetworkImpl network)
    {
        double[][][] copy = new double[network.getLayers()
                                              .size()][][];
        for (int layerIndex = 0; layerIndex < copy.length; layerIndex++)
        {
            double[][] weights = network.getLayers()
                                        .get(layerIndex)
                                        .getWeights();
            copy[layerIndex] = new double[weights.length][];
            for (int neuronIndex = 0; neuronIndex < weights.length; neuronIndex++)
            {
                copy[layerIndex][neuronIndex] = weights[neuronIndex].clone();
            }
        }
        return copy;
    }

    private static double[][] copyBiases(NeuralNetworkImpl network)
    {
        double[][] copy = new double[network.getLayers()
                                            .size()][];
        for (int layerIndex = 0; layerIndex < copy.length; layerIndex++)
        {
            copy[layerIndex] = network.getLayers()
                                      .get(layerIndex)
                                      .getBiases()
                                      .clone();
        }
        return copy;
    }

    @FunctionalInterface
    private static interface ParameterWriter
    {
        public void write(double value);
    }
}
