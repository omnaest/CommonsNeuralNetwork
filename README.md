# CommonsNeuralNetwork

A small feedforward neural network with backpropagation training, in plain Java 17 with **no runtime dependencies** — the arithmetic is `double[]` and nothing else.

Built to be read. The whole thing is a facade, a handful of domain interfaces, and three internal classes.

```java
NeuralNetwork network = NeuralNetworkUtils.newNetwork()
                                          .withInputSize(2)
                                          .addLayer(4, DefaultActivationFunction.TANH)
                                          .addLayer(1, DefaultActivationFunction.SIGMOID)
                                          .build();

network.train(trainingSet, TrainingConfiguration.builder()
                                                .learningRate(0.5)
                                                .epochs(4000)
                                                .build());

double[] output = network.predict(new double[] { 1, 0 });
```

## Design notes worth knowing

- **`derivativeOfActivated(double)`** — the activation derivative is expressed in terms of the already-activated value (`y*(1-y)`, `1-y²`, …) rather than the weighted input. Every common activation differentiates that way and the forward pass already holds that value, so no implementation has to cache anything.
- **`Layer.backward`** reads each weight into the outgoing gradient *before* updating it — the incoming gradient must use the weights as they stood during the forward pass. That ordering is the classic off-by-one-statement bug in hand-written backpropagation.
- **Seeding defaults to a fixed seed**, not system entropy, so runs are reproducible unless you opt out.
- `NeuralNetwork` is stateful and not thread-safe: layers cache the forward pass for the backward pass.

## Tests

`BackpropagationGradientTest` is the one that earns its keep — it checks every analytic weight and bias partial derivative against a central difference of the loss, on a deliberately asymmetric topology so an index swap between the neuron and input dimensions cannot hide. It extracts the analytic gradient by running the real training step at `learningRate = 1.0`, where `weightBefore − weightAfter` *is* the gradient, so no accessor exists purely for testing.

The rest: XOR convergence (not linearly separable, so it can only be solved via a learned hidden representation), a linear-regression case with analytically known weights, topology validation, and seed reproducibility.

## Build

```
mvn clean install
```
