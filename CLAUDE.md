# CommonsNeuralNetwork

Small feedforward neural network with backpropagation training (`org.omnaest.utils.neuralnetwork`). No runtime dependencies — plain `double[]` math, Java 17.

## Build

```cmd
mvn clean install
mvn test -Dtest=NeuralNetworkUtilsTest#testLearnsExclusiveOr
```

## Architecture

Static facade `NeuralNetworkUtils` returning a `NeuralNetworkBuilder`. Domain interfaces live in `neuralnetwork.domain`; implementations in `neuralnetwork.internal`.

Training is online stochastic gradient descent: one forward pass and one backward pass per example, weights updated immediately. Each `Layer` caches the input and activation of its last forward pass because the backward pass needs both — which is why a `NeuralNetwork` instance is **stateful and not thread safe**.

## Package map

| Package | What lives here |
|---|---|
| `org.omnaest.utils.neuralnetwork` | `NeuralNetworkUtils` facade |
| `neuralnetwork.domain` | `NeuralNetwork`, `NeuralNetworkBuilder`, `ActivationFunction`, `LossFunction`, `TrainingExample`, `TrainingSet`, `TrainingConfiguration`, `TrainingResult` and the `Default*` enums |
| `neuralnetwork.internal` | `Layer`, `NeuralNetworkImpl`, `NeuralNetworkBuilderImpl` |

## Key classes

- **`NeuralNetworkUtils.newNetwork()`** — entry point; declare `withInputSize(..)` then `addLayer(..)` in signal flow order, last layer is the output layer
- **`ActivationFunction`** — the extension point; `derivativeOfActivated(..)` is expressed in terms of the *already activated* value, not the weighted input, because every common activation differentiates that way and the forward pass has that value at hand. Built-ins: `DefaultActivationFunction.{IDENTITY,SIGMOID,TANH,RELU}`
- **`LossFunction`** — supplies both the loss and the gradient that backpropagation seeds from. Built-in: `DefaultLossFunction.MEAN_SQUARED_ERROR`
- **`Layer.backward(gradient, learningRate)`** — updates this layer and returns the gradient for the layer before it. Reads each weight into the outgoing gradient *before* updating it, since the incoming gradient must use the weights as they were during the forward pass
- **`NeuralNetworkImpl.applyBackpropagation(..)`** — one SGD step; public within `internal` so the gradient test exercises the same code path training uses

## Determinism

`withRandomSeed(..)` defaults to a **fixed** seed, not to system entropy — weight initialisation and per-epoch shuffling both draw from that one `Random`, so an unseeded network still trains reproducibly. Tests rely on this.

## Tests

- `NeuralNetworkUtilsTest` — XOR convergence (not linearly separable, so it can only be solved via a learned hidden representation), a linear-regression case with analytically known weights, topology/validation, seed reproducibility
- `BackpropagationGradientTest` — verifies every analytic weight and bias partial derivative against a central-difference of the loss. Extracts the analytic gradient by running `applyBackpropagation` with `learningRate = 1.0`, where `weightBefore - weightAfter` *is* the gradient — no gradient accessor is exposed for testing alone. Topology is deliberately asymmetric (3→4→5→2) so an index swap between the neuron and input dimension cannot hide.

## Dependencies

None at compile scope. Lombok (`provided`, from `CommonsParent`) for `@Value`/`@Builder` on the value types. `CommonsTest` at test scope; tests use the JUnit 4 API via the vintage engine, matching the rest of the Commons chain.
