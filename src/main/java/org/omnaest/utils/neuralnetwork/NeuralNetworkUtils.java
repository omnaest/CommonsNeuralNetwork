package org.omnaest.utils.neuralnetwork;

import org.omnaest.utils.neuralnetwork.domain.NeuralNetworkBuilder;
import org.omnaest.utils.neuralnetwork.internal.NeuralNetworkBuilderImpl;

/**
 * Entry point of the neural network library.
 *
 * <pre>
 * NeuralNetwork network = NeuralNetworkUtils.newNetwork()
 *                                           .withInputSize(2)
 *                                           .addLayer(4, DefaultActivationFunction.TANH)
 *                                           .addLayer(1, DefaultActivationFunction.SIGMOID)
 *                                           .build();
 *
 * network.train(TrainingSet.of(TrainingExample.of(new double[] { 0, 1 }, new double[] { 1 })),
 *               TrainingConfiguration.builder()
 *                                    .learningRate(0.5)
 *                                    .epochs(3000)
 *                                    .build());
 *
 * double[] output = network.predict(new double[] { 0, 1 });
 * </pre>
 */
public class NeuralNetworkUtils
{
    private NeuralNetworkUtils()
    {
    }

    public static NeuralNetworkBuilder newNetwork()
    {
        return new NeuralNetworkBuilderImpl();
    }
}
