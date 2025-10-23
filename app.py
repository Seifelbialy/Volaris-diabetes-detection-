from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert form data to feature array
        features = []
        for feature in feature_names:
            value = data.get(feature, '0')
            try:
                features.append(float(value))
            except ValueError:
                features.append(0.0)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Get confidence (probability of predicted class)
        confidence = max(prediction_proba) * 100
        
        # Determine risk level
        if prediction == 1:
            if confidence >= 80:
                risk_level = "High Risk"
                recommendation = "Please consult with a healthcare provider immediately for diabetes screening and management."
            elif confidence >= 60:
                risk_level = "Moderate Risk"
                recommendation = "Consider scheduling a diabetes screening and discuss lifestyle modifications with your doctor."
            else:
                risk_level = "Low-Moderate Risk"
                recommendation = "Maintain healthy lifestyle habits and consider regular health checkups."
        else:
            if confidence >= 80:
                risk_level = "Low Risk"
                recommendation = "Continue maintaining your healthy lifestyle habits!"
            else:
                risk_level = "Low Risk"
                recommendation = "Keep up the good work with your health habits!"
        
        result = {
            'prediction': int(prediction),
            'confidence': round(confidence, 1),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'probabilities': {
                'no_diabetes': round(prediction_proba[0] * 100, 1),
                'diabetes': round(prediction_proba[1] * 100, 1)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

