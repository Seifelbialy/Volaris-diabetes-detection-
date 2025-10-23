import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# For this demo, we'll create a synthetic dataset with the same features
# In a real scenario, you would load your actual diabetes dataset
def create_synthetic_diabetes_data():
    """Create synthetic diabetes data for demonstration"""
    np.random.seed(42)
    n_samples = 10000
    
    # Create synthetic features based on typical diabetes risk factors
    data = {
        'HighBP': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'HighChol': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'CholCheck': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'BMI': np.random.normal(28, 6, n_samples).clip(15, 50),
        'Smoker': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Stroke': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'HeartDiseaseorAttack': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'PhysActivity': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Fruits': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Veggies': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'HvyAlcoholConsump': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'AnyHealthcare': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'NoDocbcCost': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'GenHlth': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
        'MentHlth': np.random.randint(0, 31, n_samples),
        'PhysHlth': np.random.randint(0, 31, n_samples),
        'DiffWalk': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'Sex': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'Age': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 
                               n_samples, p=[0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'Education': np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
        'Income': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_samples, p=[0.1, 0.1, 0.1, 0.2, 0.2, 0.15, 0.1, 0.05])
    }
    
    # Create target variable with some logic
    diabetes_risk = (
        data['HighBP'] * 0.3 +
        data['HighChol'] * 0.2 +
        (data['BMI'] > 30) * 0.3 +
        data['HeartDiseaseorAttack'] * 0.4 +
        (data['GenHlth'] >= 4) * 0.2 +
        (data['Age'] >= 8) * 0.3 +
        (data['PhysActivity'] == 0) * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    data['Diabetes_binary'] = (diabetes_risk > 0.5).astype(int)
    
    return pd.DataFrame(data)

# Create and train the model
print("Creating synthetic diabetes dataset...")
data = create_synthetic_diabetes_data()

print("Dataset shape:", data.shape)
print("Diabetes cases:", data['Diabetes_binary'].sum())
print("Diabetes rate:", data['Diabetes_binary'].mean())

# Prepare features and target
y = data['Diabetes_binary']
X = data.drop('Diabetes_binary', axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=10,
    n_jobs=-1
)
rf.fit(X_scaled, y)

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names for reference
feature_names = list(X.columns)
joblib.dump(feature_names, 'models/feature_names.pkl')

print("Model saved successfully!")
print("Feature names:", feature_names)

# Show feature importance
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))

