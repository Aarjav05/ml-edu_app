import 'dart:isolate';
import 'dart:typed_data';
import 'neural_network.dart';

/// Messages sent TO the training isolate
class TrainRequest {
  final List<List<double>> points; // [[x, y], ...]
  final List<double> labels;       // [0 or 1, ...]
  final Map<String, dynamic>? networkConfig; // layerSizes, activation, lr
  final int stepsPerMessage;
  final int batchSize;
  final int gridResolution;

  TrainRequest({
    required this.points,
    required this.labels,
    this.networkConfig,
    this.stepsPerMessage = 10,
    this.batchSize = 16,
    this.gridResolution = 80,
  });
}

/// Messages sent FROM the training isolate
class TrainResult {
  final Map<String, dynamic> weights;
  final Float64List? gridPredictions; // Sent every N steps
  final double loss;
  final double accuracy;
  final int epoch;
  final double stepsPerSecond;

  TrainResult({
    required this.weights,
    this.gridPredictions,
    required this.loss,
    required this.accuracy,
    required this.epoch,
    required this.stepsPerSecond,
  });
}

/// Commands to control the isolate
enum TrainCommand { start, stop, reset, updateData, updateConfig }

class TrainMessage {
  final TrainCommand command;
  final TrainRequest? request;

  TrainMessage(this.command, [this.request]);
}

/// Fallback runner for platforms where Isolate isn't supported (like some Web modes)
class _FallbackRunner {
  bool running = false;
  NeuralNetwork? nn;
  List<Float64List> inputs = [];
  List<double> targets = [];
  int epoch = 0;
  int batchSize = 16;
  int stepsPerMessage = 10;
  int gridResolution = 80;
  final Function(TrainResult) onResult;

  _FallbackRunner(this.onResult);

  void buildNetwork(Map<String, dynamic> config) {
    nn = NeuralNetwork(
      layerSizes: List<int>.from(config['layerSizes'] ?? [2, 16, 16, 1]),
      hiddenActivation: Activation.values[config['hiddenActivation'] ?? 1],
      learningRate: (config['learningRate'] ?? 0.03).toDouble(),
    );
    epoch = 0;
  }

  void setData(TrainRequest req) {
    inputs = req.points.map((p) => Float64List.fromList(p)).toList();
    targets = List<double>.from(req.labels);
    batchSize = req.batchSize;
    stepsPerMessage = req.stepsPerMessage;
    gridResolution = req.gridResolution;
  }

  Float64List computeGrid() {
    if (nn == null) return Float64List(0);
    final res = gridResolution;
    final grid = Float64List(res * res);
    for (int gy = 0; gy < res; gy++) {
      for (int gx = 0; gx < res; gx++) {
        final x = gx / (res - 1) * 2.0 - 1.0;
        final y = gy / (res - 1) * 2.0 - 1.0;
        grid[gy * res + gx] = nn!.predict(x, y);
      }
    }
    return grid;
  }

  Future<void> trainingLoop() async {
    while (running && nn != null && inputs.isNotEmpty) {
      final sw = Stopwatch()..start();

      double totalLoss = 0;
      for (int s = 0; s < stepsPerMessage; s++) {
        totalLoss += nn!.trainEpoch(inputs, targets, batchSize);
        epoch++;
      }

      sw.stop();
      final avgLoss = totalLoss / stepsPerMessage;
      final accuracy = nn!.computeAccuracy(inputs, targets);
      final elapsed = sw.elapsedMicroseconds;
      final stepsPerSec = elapsed > 0 ? stepsPerMessage / (elapsed / 1e6) : 0.0;

      final grid = computeGrid();

      onResult(TrainResult(
        weights: nn!.serialize(),
        gridPredictions: grid,
        loss: avgLoss,
        accuracy: accuracy,
        epoch: epoch,
        stepsPerSecond: stepsPerSec,
      ));

      // Yield to UI thread
      await Future.delayed(const Duration(milliseconds: 16));
    }
  }

  void handleCommand(TrainMessage message) {
    switch (message.command) {
      case TrainCommand.start:
        if (message.request != null) {
          if (nn == null && message.request!.networkConfig != null) {
            buildNetwork(message.request!.networkConfig!);
          }
          setData(message.request!);
        }
        if (!running) {
          running = true;
          trainingLoop();
        }
        break;
      case TrainCommand.stop:
        running = false;
        break;
      case TrainCommand.reset:
        running = false;
        if (message.request?.networkConfig != null) {
          buildNetwork(message.request!.networkConfig!);
        } else {
          nn?.reset();
          epoch = 0;
        }
        if (message.request != null) setData(message.request!);
        break;
      case TrainCommand.updateData:
        if (message.request != null) setData(message.request!);
        break;
      case TrainCommand.updateConfig:
        running = false;
        if (message.request?.networkConfig != null) {
          buildNetwork(message.request!.networkConfig!);
        }
        if (message.request != null) setData(message.request!);
        break;
    }
  }
}

/// Manages the training isolate lifecycle
class TrainingManager {
  Isolate? _isolate;
  ReceivePort? _receivePort;
  SendPort? _sendPort;
  _FallbackRunner? _fallbackRunner;
  bool _isTraining = false;

  bool get isTraining => _isTraining;

  Function(TrainResult)? onResult;

  /// Prefer isolate, but fallback to async loop if on Web or if spawn fails
  Future<void> start(TrainRequest request) async {
    await stop();

    // In Flutter Web, Isolate.spawn may not be supported or stable depending on compile target.
    // Using a fallback runner. Note: We use a simple try-catch to detect if Isolate works.
    try {
      _receivePort = ReceivePort();
      _isolate = await Isolate.spawn(
        _trainingEntry,
        _receivePort!.sendPort,
      );

      bool gotSendPort = false;
      _receivePort!.listen((message) {
        if (!gotSendPort && message is SendPort) {
          _sendPort = message;
          gotSendPort = true;
          _sendPort!.send(TrainMessage(TrainCommand.start, request));
          _isTraining = true;
        } else if (message is TrainResult) {
          onResult?.call(message);
        }
      });
    } catch (e) {
      // Fallback to async loop on main thread
      print("Isolate spawn failed, using fallback runner: $e");
      _isolate = null;
      _receivePort?.close();
      _receivePort = null;
      _sendPort = null;
      
      _fallbackRunner = _FallbackRunner((result) {
        onResult?.call(result);
      });
      _fallbackRunner!.handleCommand(TrainMessage(TrainCommand.start, request));
      _isTraining = true;
    }
  }

  void updateData(TrainRequest request) {
    if (_fallbackRunner != null) {
      _fallbackRunner!.handleCommand(TrainMessage(TrainCommand.updateData, request));
    } else {
      _sendPort?.send(TrainMessage(TrainCommand.updateData, request));
    }
  }

  void updateConfig(TrainRequest request) {
    if (_fallbackRunner != null) {
      _fallbackRunner!.handleCommand(TrainMessage(TrainCommand.updateConfig, request));
    } else {
      _sendPort?.send(TrainMessage(TrainCommand.updateConfig, request));
    }
  }

  void pause() {
    if (_fallbackRunner != null) {
      _fallbackRunner!.handleCommand(TrainMessage(TrainCommand.stop));
    } else {
      _sendPort?.send(TrainMessage(TrainCommand.stop));
    }
    _isTraining = false;
  }

  Future<void> stop() async {
    _isTraining = false;
    _fallbackRunner?.handleCommand(TrainMessage(TrainCommand.stop));
    _fallbackRunner = null;
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _receivePort?.close();
    _receivePort = null;
    _sendPort = null;
  }

  void resetNetwork(TrainRequest request) {
    if (_fallbackRunner != null) {
      _fallbackRunner!.handleCommand(TrainMessage(TrainCommand.reset, request));
    } else {
      _sendPort?.send(TrainMessage(TrainCommand.reset, request));
    }
  }
}

/// Entry point for the training isolate
void _trainingEntry(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  NeuralNetwork? nn;
  List<Float64List> inputs = [];
  List<double> targets = [];
  bool running = false;
  int epoch = 0;
  int batchSize = 16;
  int stepsPerMessage = 10;
  int gridResolution = 80;

  void buildNetwork(Map<String, dynamic> config) {
    nn = NeuralNetwork(
      layerSizes: List<int>.from(config['layerSizes'] ?? [2, 16, 16, 1]),
      hiddenActivation: Activation.values[config['hiddenActivation'] ?? 1],
      learningRate: (config['learningRate'] ?? 0.03).toDouble(),
    );
    epoch = 0;
  }

  void setData(TrainRequest req) {
    inputs = req.points.map((p) => Float64List.fromList(p)).toList();
    targets = List<double>.from(req.labels);
    batchSize = req.batchSize;
    stepsPerMessage = req.stepsPerMessage;
    gridResolution = req.gridResolution;
  }

  Float64List computeGrid() {
    if (nn == null) return Float64List(0);
    final res = gridResolution;
    final grid = Float64List(res * res);
    for (int gy = 0; gy < res; gy++) {
      for (int gx = 0; gx < res; gx++) {
        final x = gx / (res - 1) * 2.0 - 1.0; // map to [-1, 1]
        final y = gy / (res - 1) * 2.0 - 1.0;
        grid[gy * res + gx] = nn!.predict(x, y);
      }
    }
    return grid;
  }

  Future<void> trainingLoop() async {
    while (running && nn != null && inputs.isNotEmpty) {
      final sw = Stopwatch()..start();

      double totalLoss = 0;
      for (int s = 0; s < stepsPerMessage; s++) {
        totalLoss += nn!.trainEpoch(inputs, targets, batchSize);
        epoch++;
      }

      sw.stop();
      final avgLoss = totalLoss / stepsPerMessage;
      final accuracy = nn!.computeAccuracy(inputs, targets);
      final stepsPerSec = stepsPerMessage / (sw.elapsedMicroseconds / 1e6);

      // Compute grid predictions
      final grid = computeGrid();

      mainSendPort.send(TrainResult(
        weights: nn!.serialize(),
        gridPredictions: grid,
        loss: avgLoss,
        accuracy: accuracy,
        epoch: epoch,
        stepsPerSecond: stepsPerSec,
      ));

      // Small yield to allow receiving messages
      await Future.delayed(Duration.zero);
    }
  }

  receivePort.listen((message) {
    if (message is TrainMessage) {
      switch (message.command) {
        case TrainCommand.start:
          if (message.request != null) {
            if (nn == null && message.request!.networkConfig != null) {
              buildNetwork(message.request!.networkConfig!);
            }
            setData(message.request!);
          }
          running = true;
          trainingLoop();
          break;

        case TrainCommand.stop:
          running = false;
          break;

        case TrainCommand.reset:
          running = false;
          if (message.request?.networkConfig != null) {
            buildNetwork(message.request!.networkConfig!);
          } else {
            nn?.reset();
            epoch = 0;
          }
          if (message.request != null) {
            setData(message.request!);
          }
          break;

        case TrainCommand.updateData:
          if (message.request != null) {
            setData(message.request!);
          }
          break;

        case TrainCommand.updateConfig:
          running = false;
          if (message.request?.networkConfig != null) {
            buildNetwork(message.request!.networkConfig!);
          }
          if (message.request != null) {
            setData(message.request!);
          }
          break;
      }
    }
  });
}
