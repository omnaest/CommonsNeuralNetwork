package org.omnaest.utils.neuralnetwork.domain;

import java.util.List;

import lombok.Builder;
import lombok.Value;

/**
 * Outcome of a training run. The loss history is what tells you whether the network actually learned or merely stopped complaining.
 */
@Value
@Builder
public class TrainingResult
{
    int executedEpochs;

    /**
     * Loss over the whole training set before the first weight update.
     */
    double initialLoss;

    /**
     * Loss over the whole training set after the last epoch.
     */
    double finalLoss;

    /**
     * Loss measured at the end of each epoch, in order. Empty if no epoch was executed.
     */
    List<Double> lossPerEpoch;
}
