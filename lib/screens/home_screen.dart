import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../data/datasets.dart';
import '../ml/training_isolate.dart';
import '../painters/boundary_painter.dart';
import '../painters/boundary_image_compiler.dart';
import 'dart:math';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  final TrainingManager _manager = TrainingManager();
  
  List<DataPoint> _points = [];
  DatasetType _currentDataset = DatasetType.moons;
  
  // Model Config
  double _learningRate = 0.03;
  List<int> _layerSizes = [2, 16, 16, 1];
  int _hiddenActivation = 1; // 0=relu, 1=tanh, 2=sigmoid
  
  // UI Interaction State
  int _currentAddClass = 0;
  DataPoint? _draggedPoint;
  
  // Rendering State
  ui.Image? _boundaryImage;
  ui.Image? _prevBoundaryImage;
  late AnimationController _interpController;
  late Animation<double> _interpAnimation;
  
  // Metrics
  int _epoch = 0;
  double _loss = 0.0;
  double _accuracy = 0.0;
  double _stepsPerSec = 0.0;
  
  static const int gridResolution = 80;

  @override
  void initState() {
    super.initState();
    _loadDataset(_currentDataset);
    
    _interpController = AnimationController(
       vsync: this, 
       duration: const Duration(milliseconds: 150)
    );
    _interpAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(_interpController)
      ..addListener(() { setState(() {}); });
      
    _manager.onResult = _handleTrainingResult;
  }

  void _loadDataset(DatasetType type) {
    setState(() {
      _currentDataset = type;
      _points = DatasetGenerator.generate(type, 200);
      _epoch = 0;
      _loss = 0;
      _accuracy = 0;
      _boundaryImage = null;
      _prevBoundaryImage = null;
    });
    
    final req = _buildRequest();
    if (_manager.isTraining) {
      _manager.updateData(req);
    } else {
      _manager.resetNetwork(req);
    }
  }

  TrainRequest _buildRequest({bool includeConfig = false}) {
    return TrainRequest(
      points: _points.map((p) => [p.x, p.y]).toList(),
      labels: _points.map((p) => p.label.toDouble()).toList(),
      networkConfig: includeConfig ? {
        'layerSizes': _layerSizes,
        'hiddenActivation': _hiddenActivation,
        'learningRate': _learningRate,
      } : null,
      stepsPerMessage: 10,
      batchSize: 16,
      gridResolution: gridResolution,
    );
  }

  Future<void> _handleTrainingResult(TrainResult result) async {
    setState(() {
      _epoch = result.epoch;
      _loss = result.loss;
      _accuracy = result.accuracy;
      _stepsPerSec = result.stepsPerSecond;
    });

    if (result.gridPredictions != null) {
      final img = await BoundaryImageCompiler.compile(result.gridPredictions!, gridResolution);
      
      // Setup interpolation
      if (_boundaryImage != null) {
        _prevBoundaryImage = _boundaryImage;
      }
      _boundaryImage = img;
      
      _interpController.forward(from: 0.0);
    }
  }

  void _toggleTraining() {
    if (_manager.isTraining) {
      _manager.pause();
      setState(() {});
    } else {
      _manager.start(_buildRequest(includeConfig: true));
      setState(() {});
    }
  }

  void _resetTraining() {
    _manager.resetNetwork(_buildRequest(includeConfig: true));
    setState(() {
      _epoch = 0;
      _loss = 0;
      _accuracy = 0;
      _boundaryImage = null;
      _prevBoundaryImage = null;
    });
    if (!_manager.isTraining) {
      // Need a single forward pass without loop to show initial untrained boundary?
      // Can just leave it blank until train is hit.
    }
  }

  void _updateConfig({
    double? learningRate,
    List<int>? layerSizes,
    int? hiddenActivation,
  }) {
    setState(() {
      if (learningRate != null) _learningRate = learningRate;
      if (layerSizes != null) _layerSizes = layerSizes;
      if (hiddenActivation != null) _hiddenActivation = hiddenActivation;
    });
    _manager.updateConfig(_buildRequest(includeConfig: true));
  }

  // --- Gestures ---

  Offset _screenToData(Offset pos, Size size) {
    return Offset(
      (pos.dx / size.width) * 2.0 - 1.0,
      (pos.dy / size.height) * 2.0 - 1.0,
    );
  }

  DataPoint? _findPointNear(Offset dataPos, double threshold) {
    for (final p in _points) {
      final dx = p.x - dataPos.dx;
      final dy = p.y - dataPos.dy;
      if (sqrt(dx*dx + dy*dy) < threshold) {
        return p;
      }
    }
    return null;
  }

  void _handlePanDown(DragDownDetails details, Size size) {
    final dataPos = _screenToData(details.localPosition, size);
    _draggedPoint = _findPointNear(dataPos, 0.1);
  }

  void _handlePanUpdate(DragUpdateDetails details, Size size) {
    if (_draggedPoint != null) {
      final dataPos = _screenToData(details.localPosition, size);
      setState(() {
        _draggedPoint!.x = dataPos.dx.clamp(-1.0, 1.0);
        _draggedPoint!.y = dataPos.dy.clamp(-1.0, 1.0);
      });
      if (_manager.isTraining) {
        _manager.updateData(_buildRequest());
      }
    }
  }

  void _handlePanEnd(DragEndDetails details) {
    _draggedPoint = null;
  }

  void _handleTap(TapUpDetails details, Size size) {
    final dataPos = _screenToData(details.localPosition, size);
    setState(() {
      _points.add(DataPoint(dataPos.dx, dataPos.dy, _currentAddClass));
    });
    if (_manager.isTraining) {
      _manager.updateData(_buildRequest());
    }
  }

  void _handleTwoFingerTap() {
     setState(() {
       _currentAddClass = _currentAddClass == 0 ? 1 : 0;
     });
     ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Switched to Class $_currentAddClass'),
          duration: const Duration(milliseconds: 500),
        ),
     );
  }

  void _handleLongPressStart(LongPressStartDetails details, Size size) {
    final dataPos = _screenToData(details.localPosition, size);
    final pt = _findPointNear(dataPos, 0.15);
    if (pt != null) {
      setState(() {
        _points.remove(pt);
      });
      if (_manager.isTraining) {
        _manager.updateData(_buildRequest());
      }
    }
  }

  @override
  void dispose() {
    _manager.stop();
    _interpController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final isMobile = screenWidth < 600;

    return Scaffold(
      backgroundColor: BoundaryPainter.bgColor,
      body: Stack(
        children: [
          // 1. Canvas Layer (Full Screen)
          Positioned.fill(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final size = Size(constraints.maxWidth, constraints.maxHeight);
                return GestureDetector(
                  onPanDown: (d) => _handlePanDown(d, size),
                  onPanUpdate: (d) => _handlePanUpdate(d, size),
                  onPanEnd: _handlePanEnd,
                  onTapUp: (d) => _handleTap(d, size),
                  onLongPressStart: (d) => _handleLongPressStart(d, size),
                  onDoubleTap: () => _handleTwoFingerTap(),
                  child: Stack(
                    children: [
                      if (_prevBoundaryImage != null)
                        Positioned.fill(
                          child: CustomPaint(
                            painter: BoundaryPainter(
                              points: const [],
                              boundaryImage: _prevBoundaryImage,
                              opacity: 1.0 - _interpAnimation.value,
                            ),
                          ),
                        ),
                      Positioned.fill(
                        child: CustomPaint(
                          painter: BoundaryPainter(
                            points: _points,
                            boundaryImage: _boundaryImage,
                            opacity: _prevBoundaryImage != null ? _interpAnimation.value : 1.0,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              }
            ),
          ),
          
          // 2. Control Panel Layer
          if (isMobile)
            _buildMobileDraggablePanel()
          else
            Positioned(
              left: 20,
              top: 20,
              bottom: 20,
              width: 320,
              child: _buildControlPanel(isMobile: false),
            ),
          
          // 3. Current Class Badge
          Positioned(
            right: 20,
            top: MediaQuery.of(context).padding.top + 20, // Avoid safe area issues
            child: _buildClassBadge(),
          ),
        ],
      ),
    );
  }

  Widget _buildMobileDraggablePanel() {
    return DraggableScrollableSheet(
      initialChildSize: 0.35, // Starts at 35% height so graph is highly visible
      minChildSize: 0.15, // Can swipe down to 15% to see full graph
      maxChildSize: 0.8, // Swipe up to see all settings
      builder: (context, scrollController) {
        return _buildControlPanel(isMobile: true, scrollController: scrollController);
      },
    );
  }
  
  Widget _buildClassBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 12, height: 12,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _currentAddClass == 0 ? BoundaryPainter.class0Color : BoundaryPainter.class1Color,
            ),
          ),
          const SizedBox(width: 8),
          const Text("Double-tap to switch", style: TextStyle(color: Colors.white70, fontSize: 12)),
        ],
      ),
    );
  }

  Widget _buildControlPanel({required bool isMobile, ScrollController? scrollController}) {
    return Container(
      margin: isMobile ? EdgeInsets.zero : null,
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.65),
        borderRadius: isMobile 
            ? const BorderRadius.vertical(top: Radius.circular(24))
            : BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.5),
            blurRadius: 20,
            spreadRadius: 5,
          )
        ],
      ),
      child: ClipRRect(
        borderRadius: isMobile 
            ? const BorderRadius.vertical(top: Radius.circular(24))
            : BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ui.ImageFilter.blur(sigmaX: 12, sigmaY: 12),
          child: ListView(
            controller: scrollController,
            padding: const EdgeInsets.all(24),
            children: [
              if (isMobile)
                Center(
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 20),
                    width: 40,
                    height: 5,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.3),
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              // Header
              const Text("ML Visualizer", style: TextStyle(
                color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold,
              )),
              const SizedBox(height: 24),
              
              // Controls
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _iconBtn(Icons.play_arrow, _manager.isTraining ? Colors.grey : Colors.greenAccent, _toggleTraining),
                  _iconBtn(Icons.pause, !_manager.isTraining ? Colors.grey : Colors.orangeAccent, _manager.isTraining ? _toggleTraining : null),
                  _iconBtn(Icons.restart_alt, Colors.redAccent, _resetTraining),
                ],
              ),
              const SizedBox(height: 24),
              
              // Stats
              _buildStatRow("Epoch", _epoch.toString()),
              _buildStatRow("Loss", _loss.toStringAsFixed(4)),
              _buildStatRow("Accuracy", "${(_accuracy * 100).toStringAsFixed(1)}%"),
              _buildStatRow("Steps/sec", _stepsPerSec.toStringAsFixed(0)),
              
              const Divider(color: Colors.white24, height: 32),
              
              // Datasets
              const Text("Dataset", style: TextStyle(color: Colors.white70, fontSize: 12)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8, runSpacing: 8,
                children: DatasetType.values.map((d) => ChoiceChip(
                  label: Text(d.name.toUpperCase()),
                  selected: _currentDataset == d,
                  onSelected: (s) => _loadDataset(d),
                  backgroundColor: Colors.white10,
                  selectedColor: Colors.blueAccent.withOpacity(0.3),
                  labelStyle: TextStyle(color: _currentDataset == d ? Colors.white : Colors.white70, fontSize: 11),
                )).toList(),
              ),
              
              const SizedBox(height: 24),
              
              // Hyperparameters
              const Text("Learning Rate", style: TextStyle(color: Colors.white70, fontSize: 12)),
              Slider(
                value: (log(_learningRate) / log(10)).clamp(-4.0, -0.5), // Log scale clamped
                min: -4.0, // 0.0001
                max: -0.5, // ~0.3
                divisions: 35,
                label: _learningRate.toStringAsFixed(4),
                onChanged: (v) => _updateConfig(learningRate: pow(10, v).toDouble()),
              ),
              
              const Text("Network Size", style: TextStyle(color: Colors.white70, fontSize: 12)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8, runSpacing: 8,
                children: [
                  [2, 8, 8, 1],
                  [2, 16, 16, 1],
                  [2, 32, 32, 1],
                ].map((s) => ChoiceChip(
                  label: Text("[${s[1]}, ${s[2]}]"),
                  selected: _layerSizes[1] == s[1],
                  onSelected: (_) => _updateConfig(layerSizes: s),
                  backgroundColor: Colors.white10,
                  selectedColor: Colors.blueAccent.withOpacity(0.3),
                  labelStyle: TextStyle(color: _layerSizes[1] == s[1] ? Colors.white : Colors.white70, fontSize: 11),
                )).toList(),
              ),
              
              const SizedBox(height: 16),
              const Text("Activation", style: TextStyle(color: Colors.white70, fontSize: 12)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8, runSpacing: 8,
                children: [
                  MapEntry(0, "ReLU"),
                  MapEntry(1, "Tanh"),
                  MapEntry(2, "Sigmoid"),
                ].map((e) => ChoiceChip(
                  label: Text(e.value),
                  selected: _hiddenActivation == e.key,
                  onSelected: (_) => _updateConfig(hiddenActivation: e.key),
                  backgroundColor: Colors.white10,
                  selectedColor: Colors.blueAccent.withOpacity(0.3),
                  labelStyle: TextStyle(color: _hiddenActivation == e.key ? Colors.white : Colors.white70, fontSize: 11),
                )).toList(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 14)),
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'monospace')),
        ],
      ),
    );
  }

  Widget _iconBtn(IconData icon, Color color, VoidCallback? onPressed) {
    return Container(
      decoration: BoxDecoration(
        color: color.withOpacity(onPressed == null ? 0.05 : 0.1),
        shape: BoxShape.circle,
        border: Border.all(color: color.withOpacity(onPressed == null ? 0.1 : 0.3)),
      ),
      child: IconButton(
        icon: Icon(icon, color: onPressed == null ? Colors.white24 : color),
        onPressed: onPressed,
      ),
    );
  }
}
