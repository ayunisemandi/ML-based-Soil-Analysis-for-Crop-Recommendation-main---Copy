from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from skopt import BayesSearchCV  # Bayesian Optimization

app = Flask(__name__)
CORS(app, resources={r"/process": {"origins": "*"}})  # Allow CORS for Flutter app

# Ensure static folder exists (for confusion matrix, if needed for logging)
if not os.path.exists('static'):
    os.makedirs('static')

# Load or train the model
def load_or_train_model():
    model_path = 'model.pkl'
    minmax_path = 'minmaxscaler.pkl'
    stand_path = 'standscaler.pkl'

    # Try to load existing model and scalers
    if os.path.exists(model_path) and os.path.exists(minmax_path) and os.path.exists(stand_path):
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
        with open(minmax_path, 'rb') as f:
            mx = pickle.load(f)
        with open(stand_path, 'rb') as f:
            sc = pickle.load(f)
        return best_model, mx, sc

    # Load the dataset
    try:
        crop = pd.read_csv("Crop_recommendation.csv")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        crop = pd.DataFrame()

    # Validate numeric columns
    numeric_columns = ['N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'Temp (°C)', 'Humidity (%)', 'pH', 'Rainfall (mm)']
    for col in numeric_columns:
        if col in crop.columns:
            crop[col] = pd.to_numeric(crop[col], errors='coerce')
            if crop[col].isna().any():
                print(f"Warning: Column {col} contains invalid values, filling with median.")
                crop[col].fillna(crop[col].median(), inplace=True)
        else:
            print(f"Error: Column {col} not found in CSV.")
            crop[col] = 0  # Add missing column with default value

    # Map labels to numerical values
    crop_dict = {'ginger': 1, 'turmeric': 2, 'cinnamon': 3, 'coffee': 4}
    reverse_crop_dict = {v: k for k, v in crop_dict.items()}

    if 'label' in crop.columns:
        crop['label'] = crop['label'].map(crop_dict)
        if crop['label'].isna().any():
            print("Warning: Some labels could not be mapped, dropping invalid rows.")
            crop = crop.dropna(subset=['label'])
    else:
        print("Error: 'label' column not found in CSV.")
        crop['label'] = 1  # Default to ginger

    # Create guide lookup dictionary
    guide_lookup = {}
    if 'guide' in crop.columns and 'label' in crop.columns:
        guide_lookup = dict(zip(crop['label'].map(reverse_crop_dict), crop['guide']))
    else:
        print("Error: 'guide' or 'label' column missing, guide lookup will be empty.")
        guide_lookup = {crop_name: "No cultivation guide available." for crop_name in crop_dict.keys()}

    # Split data into features and labels
    X = crop[numeric_columns]
    y = crop['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply MinMaxScaler
    mx = MinMaxScaler()
    X_train = mx.fit_transform(X_train)
    X_test = mx.transform(X_test)

    # Apply StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialize models
    models = {
        'LogisticRegression': LogisticRegression(),
        'GaussianNB': GaussianNB(),
        'SVC': SVC(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'ExtraTreeClassifier': ExtraTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'BaggingClassifier': BaggingClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
    }

    # Evaluate models
    best_model_name = ""
    best_accuracy = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        
        print(f"{name}:")
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

    # Train the best model
    best_model = models[best_model_name]

    # Apply Bayesian Optimization only if the model supports hyperparameter tuning
    if best_model_name == "GaussianNB":
        print("Skipping Bayesian Optimization because GaussianNB does not require hyperparameter tuning.")
    else:
        param_space = {}
        
        if best_model_name == "RandomForestClassifier":
            param_space = {'n_estimators': (10, 200), 'max_depth': (5, 50)}
        elif best_model_name == "SVC":
            param_space = {'C': (0.1, 10), 'gamma': (0.01, 1)}
        elif best_model_name == "KNeighborsClassifier":
            param_space = {'n_neighbors': (1, 20)}
        elif best_model_name == "GradientBoostingClassifier":
            param_space = {'n_estimators': (50, 300), 'learning_rate': (0.01, 0.2)}

        if param_space:
            opt = BayesSearchCV(best_model, param_space, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            opt.fit(X_train, y_train)
            best_model = opt.best_estimator_
            print(f"Optimized {best_model_name} with Bayesian Optimization.")

    # Save the model and scalers
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(minmax_path, 'wb') as f:
        pickle.dump(mx, f)
    with open(stand_path, 'wb') as f:
        pickle.dump(sc, f)

    return best_model, mx, sc

# Load or train model and scalers
best_model, mx, sc = load_or_train_model()

# Function for crop recommendation with input validation
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    # Basic input validation
    if not (0 <= ph <= 14 and 0 <= humidity <= 100):
        raise ValueError("pH must be between 0 and 14, and humidity between 0 and 100.")
    
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                              columns=numeri_columns)
    mx_features = mx.transform(input_data)
    sc_mx_features = sc.transform(mx_features)
    prediction = best_model.predict(sc_mx_features)
    return prediction[0]

@app.route('/process', methods=['POST'])
def process():
    try:
        # Expect only JSON requests for Flutter
        if not request.is_json:
            return jsonify({'error': 'Only JSON requests are supported'}), 400
        
        data = request.get_json()

        # Extract and validate input values with defaults
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        # Predict crop and get the crop name
        predicted_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall)
        predicted_crop_name = reverse_crop_dict.get(predicted_crop, "Unknown Crop")

        # Lookup guide text
        guide_text = guide_lookup.get(predicted_crop_name, "No cultivation guide available.")

        # Return JSON response for Flutter
        return jsonify({
            'predicted_crop': predicted_crop_name,
            'cultivation_guide': guide_text
        })

    except (ValueError, KeyError) as e:
        error_message = f"Error: Invalid input. {str(e)}" if isinstance(e, ValueError) else "Error: Missing required fields."
        return jsonify({'error': error_message}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)