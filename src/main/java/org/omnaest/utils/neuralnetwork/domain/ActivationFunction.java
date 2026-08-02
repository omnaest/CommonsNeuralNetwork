package org.omnaest.utils.neuralnetwork.domain;

/**
 * Non linear function applied to the weighted input of a single neuron. Without it a stack of layers would collapse into a single linear mapping, so this
 * is the extension point that makes a multi layer network more expressive than a single one.
 *
 * @see DefaultActivationFunction
 */
public interface ActivationFunction
{
    public double activate(double weightedInput);

    /**
     * Derivative of {@link #activate(double)} expressed in terms of the <b>already activated</b> value rather than the weighted input.
     * <p>
     * Every commonly used activation function can be differentiated this way (sigmoid: <code>y*(1-y)</code>, tanh: <code>1-y&sup2;</code>, ...), and the
     * forward pass has the activated value at hand anyway. Phrasing the contract like this spares every implementation from caching the weighted input.
     *
     * @param activatedValue
     *            the value previously returned by {@link #activate(double)}
     */
    public double derivativeOfActivated(double activatedValue);

    public default String getName()
    {
        return this.getClass()
                   .getSimpleName();
    }
}
