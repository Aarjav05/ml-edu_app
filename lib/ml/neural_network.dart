import 'dart:math';
import 'dart:typed_data';

/// Activation function types
enum Activation { relu, tanh, sigmoid }

/// A lightweight, pure-Dart dense neural network for real-time training.
class NeuralNetwork {
  final List<int> layerSizes;
  final Activation hiddenActivation;
  late List<Float64List> weights;
  late List<Float64List> biases;
  final Random _rng = Random(42);
  double learningRate;

  NeuralNetwork({
    required this.layerSizes,
    this.hiddenActivation = Activation.tanh,
    this.learningRate = 0.03,
  }) {
    _initializeWeights();
  }

  /// Xavier initialization for weights
  void _initializeWeights() {
    weights = [];
    biases = [];
    for (int i = 0; i < layerSizes.length - 1; i++) {
      final fanIn = layerSizes[i];
      final fanOut = layerSizes[i + 1];
      final limit = sqrt(6.0 / (fanIn + fanOut));
      weights.add(Float64List.fromList(
        List.generate(fanIn * fanOut, (_) => (_rng.nextDouble() * 2 - 1) * limit),
      ));
      biases.add(Float64List(fanOut));
    }
  }

  /// Reset weights to random initialization
  void reset() {
    _initializeWeights();
  }

  /// Serialize weights and biases for Isolate transfer
  Map<String, dynamic> serialize() {
    return {
      'layerSizes': layerSizes,
      'hiddenActivation': hiddenActivation.index,
      'learningRate': learningRate,
      'weights': weights.map((w) => Float64List.fromList(w)).toList(),
      'biases': biases.map((b) => Float64List.fromList(b)).toList(),
    };
  }

  /// Deserialize weights from Isolate message
  static NeuralNetwork deserialize(Map<String, dynamic> data) {
    final nn = NeuralNetwork(
      layerSizes: List<int>.from(data['layerSizes']),
      hiddenActivation: Activation.values[data['hiddenActivation']],
      learningRate: data['learningRate'],
    );
    nn.weights = (data['weights'] as List).map((w) => Float64List.fromList(List<double>.from(w))).toList();
    nn.biases = (data['biases'] as List).map((b) => Float64List.fromList(List<double>.from(b))).toList();
    return nn;
  }

  /// Load weights from serialized data (in-place)
  void loadWeights(Map<String, dynamic> data) {
    weights = (data['weights'] as List).map((w) => Float64List.fromList(List<double>.from(w))).toList();
    biases = (data['biases'] as List).map((b) => Float64List.fromList(List<double>.from(b))).toList();
    learningRate = data['learningRate'];
  }

  // ─── Activation Functions ───

  double _activate(double x, Activation act) {
    switch (act) {
      case Activation.relu:
        return x > 0 ? x : 0.01 * x; // leaky relu
      case Activation.tanh:
        // Clamped to prevent overflow
        final cx = x.clamp(-10.0, 10.0);
        final e2x = exp(2 * cx);
        return (e2x - 1) / (e2x + 1);
      case Activation.sigmoid:
        return 1.0 / (1.0 + exp(-x.clamp(-10.0, 10.0)));
    }
  }

  double _activateDerivative(double output, Activation act) {
    switch (act) {
      case Activation.relu:
        return output > 0 ? 1.0 : 0.01;
      case Activation.tanh:
        return 1.0 - output * output;
      case Activation.sigmoid:
        return output * (1.0 - output);
    }
  }

  // ─── Forward Pass ───

  /// Returns all layer activations (including input)
  List<Float64List> _forward(Float64List input) {
    final activations = <Float64List>[input];

    for (int l = 0; l < weights.length; l++) {
      final prevSize = layerSizes[l];
      final currSize = layerSizes[l + 1];
      final prev = activations.last;
      final curr = Float64List(currSize);
      final isOutput = l == weights.length - 1;
      final act = isOutput ? Activation.sigmoid : hiddenActivation;

      for (int j = 0; j < currSize; j++) {
        double sum = biases[l][j];
        for (int i = 0; i < prevSize; i++) {
          sum += prev[i] * weights[l][i * currSize + j];
        }
        curr[j] = _activate(sum, act);
      }
      activations.add(curr);
    }
    return activations;
  }

  /// Predict a single output for a 2D input
  double predict(double x, double y) {
    final input = Float64List.fromList([x, y]);
    final acts = _forward(input);
    return acts.last[0];
  }

  /// Predict for a batch of inputs, returns list of outputs
  Float64List predictBatch(Float64List flatInputs) {
    final n = flatInputs.length ~/ 2;
    final results = Float64List(n);
    for (int i = 0; i < n; i++) {
      final input = Float64List.fromList([flatInputs[i * 2], flatInputs[i * 2 + 1]]);
      final acts = _forward(input);
      results[i] = acts.last[0];
    }
    return results;
  }

  // ─── Backpropagation ───

  /// Train on a single sample. Returns the loss.
  double _trainSingle(Float64List input, double target) {
    // Forward
    final activations = _forward(input);
    final output = activations.last[0];

    // Binary cross-entropy loss
    final clampedOut = output.clamp(1e-7, 1.0 - 1e-7);
    final loss = -(target * log(clampedOut) + (1 - target) * log(1 - clampedOut));

    // Backward — compute deltas
    final deltas = <Float64List>[];
    for (int l = weights.length - 1; l >= 0; l--) {
      final currSize = layerSizes[l + 1];
      final isOutput = l == weights.length - 1;
      final act = isOutput ? Activation.sigmoid : hiddenActivation;
      final delta = Float64List(currSize);

      if (isOutput) {
        // Output layer: dL/dz = output - target (for BCE + sigmoid)
        delta[0] = output - target;
      } else {
        // Hidden layer
        final nextSize = layerSizes[l + 2];
        final nextDelta = deltas.last;
        for (int j = 0; j < currSize; j++) {
          double sum = 0;
          for (int k = 0; k < nextSize; k++) {
            sum += nextDelta[k] * weights[l + 1][j * nextSize + k];
          }
          delta[j] = sum * _activateDerivative(activations[l + 1][j], act);
        }
      }
      deltas.add(delta);
    }

    // Reverse deltas so index matches layer
    final orderedDeltas = deltas.reversed.toList();

    // Update weights and biases
    for (int l = 0; l < weights.length; l++) {
      final prevSize = layerSizes[l];
      final currSize = layerSizes[l + 1];
      final prev = activations[l];
      final delta = orderedDeltas[l];

      for (int j = 0; j < currSize; j++) {
        for (int i = 0; i < prevSize; i++) {
          weights[l][i * currSize + j] -= learningRate * delta[j] * prev[i];
        }
        biases[l][j] -= learningRate * delta[j];
      }
    }

    return loss;
  }

  /// Train on a mini-batch. Returns average loss.
  double trainBatch(List<Float64List> inputs, List<double> targets) {
    double totalLoss = 0;
    for (int i = 0; i < inputs.length; i++) {
      totalLoss += _trainSingle(inputs[i], targets[i]);
    }
    return totalLoss / inputs.length;
  }

  /// Train one epoch over the full dataset with mini-batch SGD.
  /// Returns average loss.
  double trainEpoch(List<Float64List> inputs, List<double> targets, int batchSize) {
    // Shuffle indices
    final indices = List.generate(inputs.length, (i) => i);
    indices.shuffle(_rng);

    double totalLoss = 0;
    int batches = 0;

    for (int start = 0; start < indices.length; start += batchSize) {
      final end = min(start + batchSize, indices.length);
      final batchInputs = <Float64List>[];
      final batchTargets = <double>[];
      for (int i = start; i < end; i++) {
        batchInputs.add(inputs[indices[i]]);
        batchTargets.add(targets[indices[i]]);
      }
      totalLoss += trainBatch(batchInputs, batchTargets);
      batches++;
    }

    return totalLoss / batches;
  }

  /// Compute accuracy on the dataset
  double computeAccuracy(List<Float64List> inputs, List<double> targets) {
    int correct = 0;
    for (int i = 0; i < inputs.length; i++) {
      final pred = predict(inputs[i][0], inputs[i][1]);
      final predicted = pred >= 0.5 ? 1.0 : 0.0;
      if (predicted == targets[i]) correct++;
    }
    return correct / inputs.length;
  }
}
