import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/material.dart';

class BoundaryImageCompiler {
  static const Color class0Color = Color(0xFF00E5FF); // Cyan
  static const Color class1Color = Color(0xFFFF3D00); // Deep Orange
  
  /// Compiles a 1D array of float predictions [0, 1] into a ui.Image
  /// Uses an RGBA buffer for extreme performance on the UI thread.
  static Future<ui.Image> compile(Float64List predictions, int resolution) async {
    final pixels = Uint8List(resolution * resolution * 4);
    
    for (int i = 0; i < predictions.length; i++) {
      final p = predictions[i];
      
      final c0 = class0Color;
      final c1 = class1Color;
      
      int r = 0, g = 0, b = 0;
      int a = 200; // Opacity of the heatmap core colors
      
      if (p < 0.5) {
        // Interpolate between background and class 0
        final intensity = (0.5 - p) * 2.0; 
        r = (c0.red * intensity).toInt();
        g = (c0.green * intensity).toInt();
        b = (c0.blue * intensity).toInt();
        a = (intensity * 200).toInt(); // Fade alpha near boundary
      } else {
        // Interpolate between background and class 1
        final intensity = (p - 0.5) * 2.0;
        r = (c1.red * intensity).toInt();
        g = (c1.green * intensity).toInt();
        b = (c1.blue * intensity).toInt();
        a = (intensity * 200).toInt(); // Fade alpha near boundary
      }
      
      final offset = i * 4;
      pixels[offset] = r;
      pixels[offset + 1] = g;
      pixels[offset + 2] = b;
      pixels[offset + 3] = a;
    }
    
    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      pixels,
      resolution,
      resolution,
      ui.PixelFormat.rgba8888,
      (result) => completer.complete(result),
    );
    
    return completer.future;
  }
}
