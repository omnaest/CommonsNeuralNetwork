package org.omnaest.utils.neuralnetwork.internal;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.omnaest.utils.neuralnetwork.domain.ActivationFunction;
import org.omnaest.utils.neuralnetwork.domain.NeuralNetwork;
import org.omnaest.utils.neuralnetwork.domain.NeuralNetworkBuilder;

public class NeuralNetworkBuilderImpl implements NeuralNetworkBuilder
{
    private static final long DEFAULT_RANDOM_SEED = 0;

    private final List<LayerDefinition> layerDefinitions = new ArrayList<>();

    private int inputSize = 0;

    private long randomSeed = DEFAULT_RANDOM_SEED;

    @Override
    public NeuralNetworkBuilder withInputSize(int inputSize)
    {
        this.inputSize = inputSize;
        return this;
    }

    @Override
    public NeuralNetworkBuilder addLayer(int neuronCount, ActivationFunction activationFunction)
    {
        this.layerDefinitions.add(new LayerDefinition(neuronCount, activationFunction));
        return this;
    }

    @Override
    public NeuralNetworkBuilder withRandomSeed(long randomSeed)
    {
        this.randomSeed = randomSeed;
        return this;
    }

    @Override
    public NeuralNetwork build()
    {
        if (this.inputSize <= 0)
        {
            throw new IllegalStateException("A positive input size must be declared via withInputSize(..)");
        }
        if (this.layerDefinitions.isEmpty())
        {
            throw new IllegalStateException("At least one layer must be declared via addLayer(..)");
        }

        Random random = new Random(this.randomSeed);
        List<Layer> layers = new ArrayList<>();
        int previousLayerSize = this.inputSize;
        for (LayerDefinition layerDefinition : this.layerDefinitions)
        {
            layers.add(new Layer(previousLayerSize, layerDefinition.neuronCount(), layerDefinition.activationFunction(), random));
            previousLayerSize = layerDefinition.neuronCount();
        }
        return new NeuralNetworkImpl(this.inputSize, layers, random);
    }

    private record LayerDefinition(int neuronCount, ActivationFunction activationFunction)
    {
        private LayerDefinition
        {
            if (neuronCount <= 0)
            {
                throw new IllegalArgumentException("A layer must have at least one neuron but got " + neuronCount);
            }
            if (activationFunction == null)
            {
                throw new IllegalArgumentException("A layer requires an activation function");
            }
        }
    }
}
