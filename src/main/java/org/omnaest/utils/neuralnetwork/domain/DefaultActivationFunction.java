package org.omnaest.utils.neuralnetwork.domain;

/**
 * The built in vocabulary of {@link ActivationFunction}s. Callers needing something else implement {@link ActivationFunction} directly.
 */
public enum DefaultActivationFunction implements ActivationFunction
{
    /**
     * Passes the weighted input through unchanged. Useful for the output layer of a regression network.
     */
    IDENTITY
    {
        @Override
        public double activate(double weightedInput)
        {
            return weightedInput;
        }

        @Override
        public double derivativeOfActivated(double activatedValue)
        {
            return 1.0;
        }
    },

    /**
     * Squashes into (0,1). The classic choice for a binary output layer.
     */
    SIGMOID
    {
        @Override
        public double activate(double weightedInput)
        {
            return 1.0 / (1.0 + Math.exp(-weightedInput));
        }

        @Override
        public double derivativeOfActivated(double activatedValue)
        {
            return activatedValue * (1.0 - activatedValue);
        }
    },

    /**
     * Squashes into (-1,1). Zero centered, which usually makes hidden layers converge faster than {@link #SIGMOID}.
     */
    TANH
    {
        @Override
        public double activate(double weightedInput)
        {
            return Math.tanh(weightedInput);
        }

        @Override
        public double derivativeOfActivated(double activatedValue)
        {
            return 1.0 - activatedValue * activatedValue;
        }
    },

    /**
     * Rectified linear unit. Cheap and does not saturate for positive inputs, but a neuron pushed permanently negative stops learning.
     */
    RELU
    {
        @Override
        public double activate(double weightedInput)
        {
            return Math.max(0.0, weightedInput);
        }

        @Override
        public double derivativeOfActivated(double activatedValue)
        {
            return activatedValue > 0.0 ? 1.0 : 0.0;
        }
    };

    @Override
    public String getName()
    {
        return this.name();
    }
}
