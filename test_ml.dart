import 'dart:io';
import 'package:ml_app/ml/training_isolate.dart';
import 'package:ml_app/data/datasets.dart';

void main() async {
  print('Starting isolate test...');
  final manager = TrainingManager();
  
  manager.onResult = (result) {
    print('Result received: Epoch ${result.epoch}, Loss: ${result.loss}');
    exit(0);
  };
  
  final points = DatasetGenerator.generateMoons(10);
  final req = TrainRequest(
    points: points.map((p) => [p.x, p.y]).toList(),
    labels: points.map((p) => p.label.toDouble()).toList(),
    networkConfig: {
      'layerSizes': [2, 16, 16, 1],
      'hiddenActivation': 1,
      'learningRate': 0.03,
    },
    stepsPerMessage: 10,
    batchSize: 16,
    gridResolution: 80,
  );
  
  try {
    await manager.start(req);
    print('Manager started!');
  } catch (e, st) {
    print('Error starting manager: $e\n$st');
    exit(1);
  }
}
