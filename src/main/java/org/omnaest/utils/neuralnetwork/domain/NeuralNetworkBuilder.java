package org.omnaest.utils.neuralnetwork.domain;

/**
 * Declares the topology of a {@link NeuralNetwork}. Layers are added in signal flow order; the last one added becomes the output layer.
 */
public interface NeuralNetworkBuilder
{
    public NeuralNetworkBuilder withInputSize(int inputSize);

    /**
     * Appends a fully connected layer behind whatever was added before it.
     *
     * @param neuronCount
     *            number of neurons, which is also the size of this layer's output
     */
    public NeuralNetworkBuilder addLayer(int neuronCount, ActivationFunction activationFunction);

    /**
     * Seeds the randomness used for the initial weights and for per epoch shuffling. Two networks built with the same seed and topology behave identically,
     * which is what makes training runs reproducible. Defaults to a fixed seed rather than to system entropy, so results are repeatable unless you opt out.
     */
    public NeuralNetworkBuilder withRandomSeed(long randomSeed);

    /**
     * @throws IllegalStateException
     *             if no input size or no layer was declared
     */
    public NeuralNetwork build();
}
