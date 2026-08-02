package org.omnaest.utils.neuralnetwork;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.omnaest.utils.neuralnetwork.domain.DefaultActivationFunction;
import org.omnaest.utils.neuralnetwork.domain.DefaultLossFunction;
import org.omnaest.utils.neuralnetwork.domain.NeuralNetwork;
import org.omnaest.utils.neuralnetwork.domain.TrainingConfiguration;
import org.omnaest.utils.neuralnetwork.domain.TrainingExample;
import org.omnaest.utils.neuralnetwork.domain.TrainingResult;
import org.omnaest.utils.neuralnetwork.domain.TrainingSet;

public class NeuralNetworkTest
{
    /**
     * XOR is the canonical proof that backpropagation works: it is not linearly separable, so a network that solves it cannot be doing so by accident with a
     * single linear mapping - the hidden layer must have learned a useful intermediate representation.
     */
    @Test
    public void testLearnsExclusiveOr() throws Exception
    {
        NeuralNetwork network = NeuralNetworkUtils.newNetwork()
                                                  .withInputSize(2)
                                                  .addLayer(4, DefaultActivationFunction.TANH)
                                                  .addLayer(1, DefaultActivationFunction.SIGMOID)
                                                  .withRandomSeed(1)
                                                  .build();

        TrainingResult trainingResult = network.train(createExclusiveOrTrainingSet(), TrainingConfiguration.builder()
                                                                                                          .learningRate(0.5)
                                                                                                          .epochs(4000)
                                                                                                          .build());

        assertTrue("Training must reduce the loss, but it went from " + trainingResult.getInitialLoss() + " to " + trainingResult.getFinalLoss(),
                   trainingResult.getFinalLoss() < trainingResult.getInitialLoss());
        assertTrue("Expected the network to fit XOR, but the final loss was " + trainingResult.getFinalLoss(), trainingResult.getFinalLoss() < 0.01);
        assertEquals(4000, trainingResult.getExecutedEpochs());

        assertEquals(0.0, network.predict(new double[] { 0, 0 })[0], 0.1);
        assertEquals(1.0, network.predict(new double[] { 0, 1 })[0], 0.1);
        assertEquals(1.0, network.predict(new double[] { 1, 0 })[0], 0.1);
        assertEquals(0.0, network.predict(new double[] { 1, 1 })[0], 0.1);
    }

    /**
     * A single neuron with identity activation is plain linear regression, so the weights it converges to are analytically known. This pins down that the
     * update rule moves the parameters to the right place, not merely that some loss decreases.
     */
    @Test
    public void testLearnsLinearFunctionWithAnalyticallyKnownWeights() throws Exception
    {
        NeuralNetwork network = NeuralNetworkUtils.newNetwork()
                                                  .withInputSize(1)
                                                  .addLayer(1, DefaultActivationFunction.IDENTITY)
                                                  .build();

        // y = 2x + 1
        network.train(TrainingSet.of(TrainingExample.of(new double[] { -2 }, new double[] { -3 }),
                                     TrainingExample.of(new double[] { -1 }, new double[] { -1 }),
                                     TrainingExample.of(new double[] { 0 }, new double[] { 1 }),
                                     TrainingExample.of(new double[] { 1 }, new double[] { 3 }),
                                     TrainingExample.of(new double[] { 2 }, new double[] { 5 })),
                      TrainingConfiguration.builder()
                                           .learningRate(0.05)
                                           .epochs(2000)
                                           .build());

        assertEquals(7.0, network.predict(new double[] { 3 })[0], 0.01);
        assertEquals(-5.0, network.predict(new double[] { -3 })[0], 0.01);
    }

    @Test
    public void testTrainingRecordsOneLossPerEpoch() throws Exception
    {
        TrainingResult trainingResult = NeuralNetworkUtils.newNetwork()
                                                          .withInputSize(2)
                                                          .addLayer(3, DefaultActivationFunction.TANH)
                                                          .addLayer(1, DefaultActivationFunction.SIGMOID)
                                                          .build()
                                                          .train(createExclusiveOrTrainingSet(), TrainingConfiguration.builder()
                                                                                                                     .epochs(25)
                                                                                                                     .build());

        assertEquals(25, trainingResult.getLossPerEpoch()
                                       .size());
        assertEquals(trainingResult.getFinalLoss(), trainingResult.getLossPerEpoch()
                                                                  .get(24),
                     0.0);
    }

    @Test
    public void testSameSeedProducesIdenticallyBehavingNetworks() throws Exception
    {
        double[] input = new double[] { 0.4, -0.2 };

        assertArrayEquals(createSeededNetwork(7).predict(input), createSeededNetwork(7).predict(input), 0.0);
        assertTrue("Different seeds must yield different initial weights",
                   Math.abs(createSeededNetwork(7).predict(input)[0] - createSeededNetwork(8).predict(input)[0]) > 1e-9);
    }

    @Test
    public void testTopologyIsReportedFromTheDeclaredLayers() throws Exception
    {
        NeuralNetwork network = NeuralNetworkUtils.newNetwork()
                                                  .withInputSize(5)
                                                  .addLayer(3, DefaultActivationFunction.RELU)
                                                  .addLayer(2, DefaultActivationFunction.SIGMOID)
                                                  .build();

        assertEquals(5, network.getInputSize());
        assertEquals(2, network.getOutputSize());
        assertEquals(2, network.predict(new double[] { 1, 2, 3, 4, 5 }).length);
    }

    @Test
    public void testPredictRejectsInputOfWrongSize() throws Exception
    {
        NeuralNetwork network = createSeededNetwork(0);

        assertThrows(IllegalArgumentException.class, () -> network.predict(new double[] { 1, 2, 3 }));
    }

    @Test
    public void testBuilderRejectsIncompleteTopology() throws Exception
    {
        assertThrows(IllegalStateException.class, () -> NeuralNetworkUtils.newNetwork()
                                                                         .addLayer(2, DefaultActivationFunction.TANH)
                                                                         .build());
        assertThrows(IllegalStateException.class, () -> NeuralNetworkUtils.newNetwork()
                                                                         .withInputSize(2)
                                                                         .build());
        assertThrows(IllegalArgumentException.class, () -> NeuralNetworkUtils.newNetwork()
                                                                            .withInputSize(2)
                                                                            .addLayer(0, DefaultActivationFunction.TANH));
    }

    @Test
    public void testCalculateLossLeavesTheNetworkUnchanged() throws Exception
    {
        NeuralNetwork network = createSeededNetwork(3);
        TrainingSet trainingSet = createExclusiveOrTrainingSet();
        double[] predictionBefore = network.predict(new double[] { 1, 0 });

        network.calculateLoss(trainingSet, DefaultLossFunction.MEAN_SQUARED_ERROR);

        assertArrayEquals(predictionBefore, network.predict(new double[] { 1, 0 }), 0.0);
    }

    private static NeuralNetwork createSeededNetwork(long randomSeed)
    {
        return NeuralNetworkUtils.newNetwork()
                                 .withInputSize(2)
                                 .addLayer(4, DefaultActivationFunction.TANH)
                                 .addLayer(1, DefaultActivationFunction.SIGMOID)
                                 .withRandomSeed(randomSeed)
                                 .build();
    }

    private static TrainingSet createExclusiveOrTrainingSet()
    {
        return TrainingSet.of(TrainingExample.of(new double[] { 0, 0 }, new double[] { 0 }),
                              TrainingExample.of(new double[] { 0, 1 }, new double[] { 1 }),
                              TrainingExample.of(new double[] { 1, 0 }, new double[] { 1 }),
                              TrainingExample.of(new double[] { 1, 1 }, new double[] { 0 }));
    }
}
