import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const MLVisualizerApp());
}

class MLVisualizerApp extends StatelessWidget {
  const MLVisualizerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ML Visualizer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0D0D15),
        colorScheme: ColorScheme.dark(
          primary: Colors.blueAccent,
          secondary: Colors.cyanAccent,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}
