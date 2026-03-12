import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../data/datasets.dart';

class BoundaryPainter extends CustomPainter {
  final List<DataPoint> points;
  final ui.Image? boundaryImage;
  final double opacity;
  
  // Theme colors
  static const Color class0Color = Color(0xFF00E5FF); // Cyan
  static const Color class1Color = Color(0xFFFF3D00); // Deep Orange
  static const Color bgColor = Color(0xFF0D0D15);

  BoundaryPainter({
    required this.points,
    this.boundaryImage,
    this.opacity = 1.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Draw Background
    final bgPaint = Paint()..color = bgColor;
    canvas.drawRect(Offset.zero & size, bgPaint);

    // 2. Draw Decision Boundary Image (if available)
    if (boundaryImage != null) {
      final paint = Paint()
        ..color = Color.fromRGBO(255, 255, 255, opacity)
        ..filterQuality = FilterQuality.medium
        ..isAntiAlias = true;
        
      final src = Rect.fromLTWH(
        0, 0,
        boundaryImage!.width.toDouble(),
        boundaryImage!.height.toDouble()
      );
      final dst = Offset.zero & size;
      
      canvas.drawImageRect(boundaryImage!, src, dst, paint);
    }

    // 3. Draw Data Points
    final pointRadius = 6.0;
    
    for (final p in points) {
      // Coordinate transform: [-1, 1] -> [0, width/height]
      final cx = (p.x + 1.0) / 2.0 * size.width;
      final cy = (p.y + 1.0) / 2.0 * size.height;
      
      final baseColor = p.label == 0 ? class0Color : class1Color;
      
      // Glow effect
      final glowPaint = Paint()
        ..color = baseColor.withOpacity(0.3)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8.0);
      canvas.drawCircle(Offset(cx, cy), pointRadius * 2, glowPaint);
      
      // Core point
      final corePaint = Paint()
        ..color = baseColor
        ..style = PaintingStyle.fill;
      canvas.drawCircle(Offset(cx, cy), pointRadius, corePaint);
      
      // White outline for contrast
      final borderPaint = Paint()
        ..color = Colors.white.withOpacity(0.8)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5;
      canvas.drawCircle(Offset(cx, cy), pointRadius, borderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant BoundaryPainter oldDelegate) {
    return oldDelegate.boundaryImage != boundaryImage ||
           oldDelegate.points != points ||
           oldDelegate.opacity != opacity;
  }
}
