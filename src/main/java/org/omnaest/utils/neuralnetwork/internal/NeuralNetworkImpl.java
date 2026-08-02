package org.omnaest.utils.neuralnetwork.internal;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.omnaest.utils.neuralnetwork.domain.LossFunction;
import org.omnaest.utils.neuralnetwork.domain.NeuralNetwork;
import org.omnaest.utils.neuralnetwork.domain.TrainingConfiguration;
import org.omnaest.utils.neuralnetwork.domain.TrainingConfiguration.EpochListener;
import org.omnaest.utils.neuralnetwork.domain.TrainingExample;
import org.omnaest.utils.neuralnetwork.domain.TrainingResult;
import org.omnaest.utils.neuralnetwork.domain.TrainingSet;

public class NeuralNetworkImpl implements NeuralNetwork
{
    private final int inputSize;

    private final List<Layer> layers;

    private final Random random;

    public NeuralNetworkImpl(int inputSize, List<Layer> layers, Random random)
    {
        this.inputSize = inputSize;
        this.layers = layers;
        this.random = random;
    }

    @Override
    public double[] predict(double[] input)
    {
        if (input.length != this.inputSize)
        {
            throw new IllegalArgumentException("Network expects an input of size " + this.inputSize + " but got " + input.length);
        }

        double[] signal = input;
        for (Layer layer : this.layers)
        {
            signal = layer.forward(signal);
        }
        return signal;
    }

    @Override
    public TrainingResult train(TrainingSet trainingSet, TrainingConfiguration configuration)
    {
        LossFunction lossFunction = configuration.getLossFunction();
        List<TrainingExample> examples = new ArrayList<>(trainingSet.toList());
        List<Double> lossPerEpoch = new ArrayList<>();

        double initialLoss = this.calculateLoss(trainingSet, lossFunction);
        double loss = initialLoss;

        for (int epoch = 1; epoch <= configuration.getEpochs(); epoch++)
        {
            if (configuration.isShuffleExamplesPerEpoch())
            {
                Collections.shuffle(examples, this.random);
            }
            for (TrainingExample example : examples)
            {
                this.applyBackpropagation(example, lossFunction, configuration.getLearningRate());
            }

            loss = this.calculateLoss(trainingSet, lossFunction);
            lossPerEpoch.add(loss);

            EpochListener epochListener = configuration.getEpochListener();
            if (epochListener != null)
            {
                epochListener.onEpochCompleted(epoch, loss);
            }
        }

        return TrainingResult.builder()
                             .executedEpochs(lossPerEpoch.size())
                             .initialLoss(initialLoss)
                             .finalLoss(loss)
                             .lossPerEpoch(List.copyOf(lossPerEpoch))
                             .build();
    }

    @Override
    public double calculateLoss(TrainingSet trainingSet, LossFunction lossFunction)
    {
        if (trainingSet.size() == 0)
        {
            return 0.0;
        }
        return trainingSet.stream()
                          .mapToDouble(example -> lossFunction.calculateLoss(this.predict(example.getInput()), example.getExpectedOutput()))
                          .average()
                          .orElse(0.0);
    }

    /**
     * One forward pass followed by one backward pass for a single example, i.e. the smallest unit of stochastic gradient descent. Kept public within this
     * internal package so gradient verification exercises the very same code path training does, rather than a reimplementation of it.
     */
    public void applyBackpropagation(TrainingExample example, LossFunction lossFunction, double learningRate)
    {
        double[] predictedOutput = this.predict(example.getInput());
        double[] gradient = lossFunction.calculateGradient(predictedOutput, example.getExpectedOutput());
        for (int layerIndex = this.layers.size() - 1; layerIndex >= 0; layerIndex--)
        {
            gradient = this.layers.get(layerIndex)
                                  .backward(gradient, learningRate);
        }
    }

    @Override
    public int getInputSize()
    {
        return this.inputSize;
    }

    @Override
    public int getOutputSize()
    {
        return this.layers.get(this.layers.size() - 1)
                          .getNeuronCount();
    }

    public List<Layer> getLayers()
    {
        return this.layers;
    }
}
