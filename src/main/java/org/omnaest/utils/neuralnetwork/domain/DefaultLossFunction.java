package org.omnaest.utils.neuralnetwork.domain;

/**
 * The built in vocabulary of {@link LossFunction}s.
 */
public enum DefaultLossFunction implements LossFunction
{
    /**
     * Mean of the squared differences. Suits regression and, in a small teaching network, binary classification as well.
     */
    MEAN_SQUARED_ERROR
    {
        @Override
        public double calculateLoss(double[] predictedOutput, double[] expectedOutput)
        {
            assertSameLength(predictedOutput, expectedOutput);
            double sumOfSquaredErrors = 0.0;
            for (int outputIndex = 0; outputIndex < predictedOutput.length; outputIndex++)
            {
                double error = predictedOutput[outputIndex] - expectedOutput[outputIndex];
                sumOfSquaredErrors += error * error;
            }
            return sumOfSquaredErrors / predictedOutput.length;
        }

        @Override
        public double[] calculateGradient(double[] predictedOutput, double[] expectedOutput)
        {
            assertSameLength(predictedOutput, expectedOutput);
            double[] gradient = new double[predictedOutput.length];
            for (int outputIndex = 0; outputIndex < predictedOutput.length; outputIndex++)
            {
                gradient[outputIndex] = 2.0 * (predictedOutput[outputIndex] - expectedOutput[outputIndex]) / predictedOutput.length;
            }
            return gradient;
        }
    };

    @Override
    public String getName()
    {
        return this.name();
    }

    private static void assertSameLength(double[] predictedOutput, double[] expectedOutput)
    {
        if (predictedOutput.length != expectedOutput.length)
        {
            throw new IllegalArgumentException("Predicted output of size " + predictedOutput.length + " cannot be compared to an expected output of size "
                    + expectedOutput.length);
        }
    }
}
