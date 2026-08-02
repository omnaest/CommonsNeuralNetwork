package org.omnaest.utils.neuralnetwork.domain;

import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

/**
 * An immutable collection of {@link TrainingExample}s presented to a network during training.
 */
@FunctionalInterface
public interface TrainingSet extends Iterable<TrainingExample>
{
    public List<TrainingExample> toList();

    @Override
    public default Iterator<TrainingExample> iterator()
    {
        return this.toList()
                   .iterator();
    }

    public default Stream<TrainingExample> stream()
    {
        return this.toList()
                   .stream();
    }

    public default int size()
    {
        return this.toList()
                   .size();
    }

    public static TrainingSet of(TrainingExample... examples)
    {
        return TrainingSet.of(Arrays.asList(examples));
    }

    public static TrainingSet of(Collection<TrainingExample> examples)
    {
        List<TrainingExample> immutableExamples = List.copyOf(examples);
        return () -> immutableExamples;
    }
}
