import 'dart:math';

class DataPoint {
  double x;
  double y;
  int label; // 0 or 1

  DataPoint(this.x, this.y, this.label);

  List<double> toList() => [x, y];
}

enum DatasetType { moons, circles, xor, spiral, gaussian }

class DatasetGenerator {
  static final Random _rng = Random(42);

  static List<DataPoint> generate(DatasetType type, int n) {
    switch (type) {
      case DatasetType.moons:
        return generateMoons(n);
      case DatasetType.circles:
        return generateCircles(n);
      case DatasetType.xor:
        return generateXOR(n);
      case DatasetType.spiral:
        return generateSpiral(n);
      case DatasetType.gaussian:
        return generateGaussian(n);
    }
  }

  /// Two interleaving half-circles
  static List<DataPoint> generateMoons(int n) {
    final points = <DataPoint>[];
    final half = n ~/ 2;
    for (int i = 0; i < half; i++) {
      final angle = pi * i / half;
      final noise = (_rng.nextDouble() - 0.5) * 0.15;
      points.add(DataPoint(
        cos(angle) * 0.5 + noise,
        sin(angle) * 0.5 + noise,
        0,
      ));
    }
    for (int i = 0; i < n - half; i++) {
      final angle = pi * i / (n - half);
      final noise = (_rng.nextDouble() - 0.5) * 0.15;
      points.add(DataPoint(
        0.5 - cos(angle) * 0.5 + noise,
        -sin(angle) * 0.5 + 0.25 + noise,
        1,
      ));
    }
    return points;
  }

  /// Concentric rings
  static List<DataPoint> generateCircles(int n) {
    final points = <DataPoint>[];
    final half = n ~/ 2;
    for (int i = 0; i < half; i++) {
      final angle = 2 * pi * i / half;
      final r = 0.25 + (_rng.nextDouble() - 0.5) * 0.08;
      points.add(DataPoint(cos(angle) * r, sin(angle) * r, 0));
    }
    for (int i = 0; i < n - half; i++) {
      final angle = 2 * pi * i / (n - half);
      final r = 0.6 + (_rng.nextDouble() - 0.5) * 0.08;
      points.add(DataPoint(cos(angle) * r, sin(angle) * r, 1));
    }
    return points;
  }

  /// Four-quadrant XOR pattern
  static List<DataPoint> generateXOR(int n) {
    final points = <DataPoint>[];
    final perQuad = n ~/ 4;
    for (int q = 0; q < 4; q++) {
      final cx = (q % 2 == 0) ? -0.4 : 0.4;
      final cy = (q < 2) ? -0.4 : 0.4;
      final label = ((q == 0) || (q == 3)) ? 0 : 1;
      final count = q < 3 ? perQuad : n - 3 * perQuad;
      for (int i = 0; i < count; i++) {
        points.add(DataPoint(
          cx + (_rng.nextDouble() - 0.5) * 0.35,
          cy + (_rng.nextDouble() - 0.5) * 0.35,
          label,
        ));
      }
    }
    return points;
  }

  /// Two interleaving spirals
  static List<DataPoint> generateSpiral(int n) {
    final points = <DataPoint>[];
    final half = n ~/ 2;
    for (int i = 0; i < half; i++) {
      final t = i / half * 2 * pi;
      final r = t / (2 * pi) * 0.6;
      final noise = (_rng.nextDouble() - 0.5) * 0.06;
      points.add(DataPoint(
        r * cos(t) + noise,
        r * sin(t) + noise,
        0,
      ));
    }
    for (int i = 0; i < n - half; i++) {
      final t = i / (n - half) * 2 * pi;
      final r = t / (2 * pi) * 0.6;
      final noise = (_rng.nextDouble() - 0.5) * 0.06;
      points.add(DataPoint(
        r * cos(t + pi) + noise,
        r * sin(t + pi) + noise,
        1,
      ));
    }
    return points;
  }

  /// Two Gaussian blobs
  static List<DataPoint> generateGaussian(int n) {
    final points = <DataPoint>[];
    final half = n ~/ 2;
    for (int i = 0; i < half; i++) {
      points.add(DataPoint(
        -0.35 + _gaussian() * 0.18,
        0.0 + _gaussian() * 0.18,
        0,
      ));
    }
    for (int i = 0; i < n - half; i++) {
      points.add(DataPoint(
        0.35 + _gaussian() * 0.18,
        0.0 + _gaussian() * 0.18,
        1,
      ));
    }
    return points;
  }

  /// Box-Muller transform for Gaussian random numbers
  static double _gaussian() {
    final u1 = _rng.nextDouble();
    final u2 = _rng.nextDouble();
    return sqrt(-2 * log(u1)) * cos(2 * pi * u2);
  }
}
