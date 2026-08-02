package org.omnaest.utils.neuralnetwork.domain;

/**
 * A fully connected feedforward network trained by backpropagation.
 * <p>
 * Instances are <b>stateful and not thread safe</b>: {@link #train(TrainingSet, TrainingConfiguration)} mutates the weights, and the forward pass caches
 * intermediate activations for the backward pass. Confine one instance to one thread.
 *
 * @see org.omnaest.utils.neuralnetwork.NeuralNetworkUtils#newNetwork()
 */
public interface NeuralNetwork
{
    /**
     * Runs the input through the network and returns the output layer activations.
     *
     * @throws IllegalArgumentException
     *             if the input length does not match {@link #getInputSize()}
     */
    public double[] predict(double[] input);

    public TrainingResult train(TrainingSet trainingSet, TrainingConfiguration configuration);

    public default TrainingResult train(TrainingSet trainingSet)
    {
        return this.train(trainingSet, TrainingConfiguration.defaults());
    }

    /**
     * Average loss over the given set without changing any weight. Use it to score a held out set against the training loss.
     */
    public double calculateLoss(TrainingSet trainingSet, LossFunction lossFunction);

    public int getInputSize();

    public int getOutputSize();
}
