package org.omnaest.utils.neuralnetwork.domain;

import lombok.Builder;
import lombok.Value;

/**
 * A single supervised training sample: one input vector together with the output the network is supposed to produce for it.
 */
@Value
@Builder
public class TrainingExample
{
    double[] input;

    double[] expectedOutput;

    public static TrainingExample of(double[] input, double[] expectedOutput)
    {
        return TrainingExample.builder()
                              .input(input)
                              .expectedOutput(expectedOutput)
                              .build();
    }
}
