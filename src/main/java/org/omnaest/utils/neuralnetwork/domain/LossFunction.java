package org.omnaest.utils.neuralnetwork.domain;

/**
 * Measures how far a predicted output is from the expected one, and supplies the gradient that backpropagation starts from.
 *
 * @see DefaultLossFunction
 */
public interface LossFunction
{
    public double calculateLoss(double[] predictedOutput, double[] expectedOutput);

    /**
     * Partial derivative of {@link #calculateLoss(double[], double[])} with respect to each element of the predicted output. This is the seed value that is
     * fed backwards through the layers.
     */
    public double[] calculateGradient(double[] predictedOutput, double[] expectedOutput);

    public default String getName()
    {
        return this.getClass()
                   .getSimpleName();
    }
}
