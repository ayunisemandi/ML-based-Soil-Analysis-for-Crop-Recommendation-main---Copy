import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:crop_recommendation_app/main.dart';

void main() {
  testWidgets('Crop Recommendation form smoke test', (
    WidgetTester tester,
  ) async {
    // Build the app and trigger a frame.
    await tester.pumpWidget(const CropRecommendationApp());

    // Verify initial UI elements
    expect(
      find.byType(TextFormField),
      findsNWidgets(7),
    ); // 7 input fields (N, P, K, temp, humidity, pH, rainfall)
    expect(find.byType(ElevatedButton), findsOneWidget); // Submit button
    expect(
      find.text('Predicted Crop: '),
      findsNothing,
    ); // No prediction initially
    expect(
      find.text('Cultivation Guide: '),
      findsNothing,
    ); // No guide initially

    // Simulate entering form data
    await tester.enterText(find.byType(TextFormField).at(0), '90'); // N
    await tester.enterText(find.byType(TextFormField).at(1), '42'); // P
    await tester.enterText(find.byType(TextFormField).at(2), '43'); // K
    await tester.enterText(
      find.byType(TextFormField).at(3),
      '20.8',
    ); // Temperature
    await tester.enterText(find.byType(TextFormField).at(4), '82'); // Humidity
    await tester.enterText(find.byType(TextFormField).at(5), '6.5'); // pH
    await tester.enterText(
      find.byType(TextFormField).at(6),
      '202.9',
    ); // Rainfall

    // Trigger form submission
    await tester.tap(find.byType(ElevatedButton));
    await tester.pump(); // Rebuild the widget after the tap

    // Since we can't mock the API easily here, we'll assume the submission triggers a state change.
    // For a real test, you'd mock the HTTP response (see below for advanced setup).
    // For now, verify that the loading state or error state might appear.
    expect(
      find.byType(CircularProgressIndicator),
      findsOneWidget,
    ); // Check if loading appears

    // Wait for the pump delay to simulate API response (adjust based on your app's delay)
    await tester.pumpAndSettle();

    // Verify the result (mocked or real, depending on your app's state management)
    // Assuming the app updates with a predicted crop after submission
    expect(
      find.text('Predicted Crop: ginger'),
      findsOneWidget,
    ); // Example prediction
    expect(
      find.textContaining('Cultivation Guide: Ensure'),
      findsOneWidget,
    ); // Example guide
  });
}
