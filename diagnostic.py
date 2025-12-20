""" 
Docstring for diagnostic
"""
import os
import sys
import json

print("API Diagnostic Check")
print("="*60)

# 1. Check if files exist
print("\n1. File Check:")
model_path = "models/logistic_champion_fixed.joblib"
features_path = "models/feature_names.json"

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Feature names file exists: {os.path.exists(features_path)}")

if os.path.exists(features_path):
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"Features in file: {features}")
    print(f"Number of features: {len(features)}")

# 2. Check environment variables
print("\n2. Environment Variables:")
print(f"MODEL_LOCAL_PATH: {os.getenv('MODEL_LOCAL_PATH', 'Not set')}")
print(f"MODEL_FEATURES_PATH: {os.getenv('MODEL_FEATURES_PATH', 'Not set')}")

# 3. Check model
print("\n3. Model Check:")
try:
    import joblib
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")

    # Check model attributes
    if hasattr(model, 'n_features_in_'):
        print(f"  n_features_in_: {model.n_features_in_}")
    else:
        print(f"  n_features_in_: Not found")

    if hasattr(model, 'feature_names_in_'):
        print(f"  feature_names_in_: {model.feature_names_in_}")
    else:
        print(f"  feature_names_in_: Not found")

    if hasattr(model, 'coef_'):
        print(f"  coef_ shape: {model.coef_.shape}")
        print(f"  coefficients: {model.coef_}")

    if hasattr(model, 'intercept_'):
        print(f"  intercept_: {model.intercept_}")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

# 4. Test prediction directly
print("\n4. Direct Prediction Test:")
try:
    import numpy as np

    # Create test data - make sure it matches what model expects
    test_data = np.array([[2019.0, 6.0]])  # Year_mean, Month_mean

    print(f"Test data shape: {test_data.shape}")
    print(f"Test data: {test_data}")

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(test_data)
        prediction = model.predict(test_data)
        print(f"✓ Direct prediction successful!")
        print(f"  Probability: {proba[0, 1]:.4f}")
        print(f"  Prediction: {prediction[0]}")
        print(f"  Risk: {'HIGH' if prediction[0] == 1 else 'LOW'}")
    else:
        prediction = model.predict(test_data)
        print(f"✓ Direct prediction successful!")
        print(f"  Prediction: {prediction[0]}")

except Exception as e:
    print(f"✗ Direct prediction failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Test the Predictor class
print("\n5. Predictor Class Test:")
try:
    # Temporarily set environment variables
    os.environ["MODEL_LOCAL_PATH"] = model_path
    os.environ["MODEL_FEATURES_PATH"] = features_path

    # Import and test
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.api.predictor import Predictor

    predictor = Predictor()
    predictor.load()

    print(f"✓ Predictor loaded successfully")
    print(f"  Source: {predictor.source}")
    print(f"  Features: {predictor.features}")
    print(f"  n_features_expected: {predictor.n_features_expected}")

    # Test with sample data
    test_features = {"Year_mean": 2019.0, "Month_mean": 6.0}
    print(f"\n  Testing with features: {test_features}")

    result = predictor.predict(test_features)
    print(f"  ✓ Predictor prediction successful!")
    print(f"    Probability: {result.get('probability')}")
    print(f"    Predicted Class: {result.get('predicted_class')}")

except ImportError as e:
    print(f"✗ Cannot import predictor: {e}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Python path: {sys.path}")
except Exception as e:
    print(f"✗ Predictor test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Diagnostic complete!")
