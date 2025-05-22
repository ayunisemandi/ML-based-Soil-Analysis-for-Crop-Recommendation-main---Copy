import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async'; // For timeout handling

void main() {
  runApp(const CropRecommendationApp());
}

class CropRecommendationApp extends StatelessWidget {
  const CropRecommendationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Crop Recommendation',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ), // Changed to green for a more agricultural theme
      home: const CropFormPage(),
    );
  }
}

class CropFormPage extends StatefulWidget {
  const CropFormPage({super.key});

  @override
  CropFormPageState createState() => CropFormPageState(); // Made public
}

class CropFormPageState extends State<CropFormPage> {
  final _formKey = GlobalKey<FormState>();
  final _nController = TextEditingController();
  final _pController = TextEditingController();
  final _kController = TextEditingController();
  final _tempController = TextEditingController();
  final _humidityController = TextEditingController();
  final _phController = TextEditingController();
  final _rainfallController = TextEditingController();
  String _predictedCrop = '';
  String _cultivationGuide = '';
  String _error = '';
  bool _isLoading = false;

  Future<void> _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _error = '';
        _predictedCrop = '';
        _cultivationGuide = '';
      });

      try {
        final response = await http
            .post(
              Uri.parse(
                'http://10.0.2.2:5000/process',
              ), // Replace with your backend IP
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({
                'N': _nController.text,
                'P': _pController.text,
                'K': _kController.text,
                'temperature': _tempController.text,
                'humidity': _humidityController.text,
                'ph': _phController.text,
                'rainfall': _rainfallController.text,
              }),
            )
            .timeout(const Duration(seconds: 10)); // Added timeout

        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          setState(() {
            _predictedCrop = data['predicted_crop'] ?? 'Unknown Crop';
            _cultivationGuide =
                data['cultivation_guide'] ?? 'No guide available';
            _error = '';
          });
        } else {
          final data = jsonDecode(response.body);
          setState(() {
            _error = data['error'] ?? 'Server error: ${response.statusCode}';
            _predictedCrop = '';
            _cultivationGuide = '';
          });
        }
      } on TimeoutException {
        setState(() {
          _error = 'Connection timed out. Please check your network.';
          _predictedCrop = '';
          _cultivationGuide = '';
        });
      } catch (e) {
        setState(() {
          _error = 'Failed to connect: $e';
          _predictedCrop = '';
          _cultivationGuide = '';
        });
      } finally {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Recommendation'),
        centerTitle: true, // Center the title for better appearance
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextFormField(
                  controller: _nController,
                  decoration: const InputDecoration(
                    labelText: 'Nitrogen (kg/ha)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 300) {
                      return 'Nitrogen must be between 0 and 300';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _pController,
                  decoration: const InputDecoration(
                    labelText: 'Phosphorus (kg/ha)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 300) {
                      return 'Phosphorus must be between 0 and 300';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _kController,
                  decoration: const InputDecoration(
                    labelText: 'Potassium (kg/ha)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 300) {
                      return 'Potassium must be between 0 and 300';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _tempController,
                  decoration: const InputDecoration(
                    labelText: 'Temperature (°C)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < -10 || numValue > 50) {
                      return 'Temperature must be between -10 and 50';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _humidityController,
                  decoration: const InputDecoration(
                    labelText: 'Humidity (%)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 100) {
                      return 'Humidity must be between 0 and 100';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _phController,
                  decoration: const InputDecoration(
                    labelText: 'pH',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 14) {
                      return 'pH must be between 0 and 14';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 12),
                TextFormField(
                  controller: _rainfallController,
                  decoration: const InputDecoration(
                    labelText: 'Rainfall (mm)',
                    border: OutlineInputBorder(),
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) return 'Enter a value';
                    final numValue = double.tryParse(value);
                    if (numValue == null) return 'Enter a valid number';
                    if (numValue < 0 || numValue > 5000) {
                      return 'Rainfall must be between 0 and 5000';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 20),
                _isLoading
                    ? const Center(child: CircularProgressIndicator())
                    : ElevatedButton(
                      onPressed: _submitForm,
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(
                          double.infinity,
                          50,
                        ), // Full-width button
                      ),
                      child: const Text('Submit'),
                    ),
                const SizedBox(height: 20),
                if (_error.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8.0),
                    child: Text(
                      _error,
                      style: const TextStyle(color: Colors.red, fontSize: 14),
                    ),
                  ),
                if (_predictedCrop.isNotEmpty)
                  Card(
                    elevation: 4,
                    margin: const EdgeInsets.only(top: 16.0),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Predicted Crop: $_predictedCrop',
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.green,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Cultivation Guide: $_cultivationGuide',
                            style: const TextStyle(fontSize: 16),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _nController.dispose();
    _pController.dispose();
    _kController.dispose();
    _tempController.dispose();
    _humidityController.dispose();
    _phController.dispose();
    _rainfallController.dispose();
    super.dispose();
  }
}
