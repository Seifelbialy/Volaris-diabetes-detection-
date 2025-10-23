# Diabetes Risk Assessment Flask App

A web application that predicts diabetes risk based on health indicators using machine learning.

## Features

- Interactive MCQ-style questionnaire
- Real-time diabetes risk prediction
- Confidence scores and probability breakdown
- Personalized health recommendations
- Modern, responsive UI
- BMI calculation guidance

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and Save the Model**
   ```bash
   python train_and_save_model.py
   ```

3. **Run the Flask Application**
   ```bash
   python app.py
   ```

4. **Access the Application**
   Open your browser and go to: `http://localhost:5000`

## Model Features

The model uses the following health indicators for prediction:

- High Blood Pressure
- High Cholesterol
- Cholesterol Check History
- BMI (Body Mass Index)
- Smoking Status
- Stroke History
- Heart Disease/Attack History
- Physical Activity
- Fruit Consumption
- Vegetable Consumption
- Heavy Alcohol Consumption
- Healthcare Coverage
- Cost Barriers to Healthcare
- General Health Rating
- Mental Health Days
- Physical Health Days
- Walking Difficulty
- Sex
- Age Group
- Education Level
- Income Level

## BMI Calculation

The app includes a BMI calculator note:
- **Formula**: BMI = Weight (kg) ÷ Height (m)²
- **Alternative**: BMI = (Weight in pounds × 703) ÷ (Height in inches)²
- **Categories**:
  - Underweight: < 18.5
  - Normal: 18.5 - 24.9
  - Overweight: 25 - 29.9
  - Obese: ≥ 30

## Risk Assessment

The model provides:
- **Prediction**: Diabetes risk (Yes/No)
- **Confidence**: Model confidence percentage
- **Risk Level**: High/Moderate/Low risk classification
- **Probabilities**: Detailed probability breakdown
- **Recommendations**: Personalized health advice

## Technical Details

- **Model**: Random Forest Classifier
- **Features**: 21 health indicators
- **Preprocessing**: StandardScaler for feature normalization
- **Framework**: Flask web framework
- **UI**: Bootstrap 5 with custom styling

## Disclaimer

This application is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with a healthcare provider for medical concerns.

