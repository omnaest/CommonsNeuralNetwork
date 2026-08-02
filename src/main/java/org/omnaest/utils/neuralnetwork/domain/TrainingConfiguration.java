package org.omnaest.utils.neuralnetwork.domain;

import lombok.Builder;
import lombok.Value;

/**
 * The knobs of one training run. Separate from the network itself so the same topology can be retrained under different settings.
 */
@Value
@Builder(toBuilder = true)
public class TrainingConfiguration
{
    /**
     * Step size of a single weight update. Too small and training crawls, too large and the loss oscillates instead of settling.
     */
    @Builder.Default
    double learningRate = 0.5;

    /**
     * Number of full passes over the training set.
     */
    @Builder.Default
    int epochs = 1000;

    @Builder.Default
    LossFunction lossFunction = DefaultLossFunction.MEAN_SQUARED_ERROR;

    /**
     * Reorders the examples before every epoch so the network does not learn the presentation order along with the data. Uses the network's own seeded
     * randomness, so a seeded network trains reproducibly either way.
     */
    @Builder.Default
    boolean shuffleExamplesPerEpoch = true;

    /**
     * Optional hook invoked after every epoch, e.g. to log progress or implement an early stop. May be <code>null</code>.
     */
    EpochListener epochListener;

    public static TrainingConfiguration defaults()
    {
        return TrainingConfiguration.builder()
                                    .build();
    }

    @FunctionalInterface
    public static interface EpochListener
    {
        public void onEpochCompleted(int epoch, double loss);
    }
}
