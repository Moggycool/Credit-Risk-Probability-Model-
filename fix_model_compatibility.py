"""
Fix model compatibility issues by loading and re-saving with current sklearn version.
"""

import joblib
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import os


def fix_model_compatibility():
    """Fix model compatibility issues."""

    model_files = [
        "models/logistic_champion_fixed.joblib",
        "models/logistic_best.pkl",
        "models/random_forest_best.pkl"
    ]

    for model_path in model_files:
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
            continue

        print(f"\nProcessing: {model_path}")

        try:
            # Try to load
            if model_path.endswith('.joblib'):
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

            print(f"  Model type: {type(model)}")

            # Check if it's a valid model
            if hasattr(model, 'predict'):
                # Test prediction
                X_test = np.array([[2019.0, 8.0]])

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    print(f"  Test prediction probability: {proba[0, 1]}")
                else:
                    pred = model.predict(X_test)
                    print(f"  Test prediction: {pred}")

                # Re-save with current sklearn version
                new_path = model_path.replace(
                    '.joblib', '_fixed.joblib').replace('.pkl', '_fixed.joblib')
                joblib.dump(model, new_path)
                print(f"  Re-saved as: {new_path}")

            else:
                print(f"  Not a valid sklearn model")

        except Exception as e:
            print(f"  Error: {e}")


def create_simple_model():
    """Create a simple working model for testing."""
    print("\nCreating simple test model...")

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000

    # Features: Year_mean (2018-2020), Month_mean (1-12)
    X = np.column_stack([
        np.random.uniform(2018, 2020, n_samples),
        np.random.uniform(1, 12, n_samples)
    ])

    # Target: Higher risk for more recent transactions
    # Simple rule: risk = 1 if Year > 2018.5 and Month > 6
    y = ((X[:, 0] > 2018.5) & (X[:, 1] > 6)).astype(int)

    # Add some noise
    noise = np.random.rand(n_samples) < 0.1
    y[noise] = 1 - y[noise]

    print(f"Training data shape: {X.shape}")
    print(f"Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    print(f"\nModel trained:")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_}")

    # Test predictions
    test_cases = [
        [2018.0, 1.0],   # Low risk
        [2018.5, 6.5],   # Medium risk
        [2019.0, 8.0],   # High risk
        [2019.0, 12.0],  # Very high risk
    ]

    print("\nTest predictions:")
    for X_test in test_cases:
        proba = model.predict_proba([X_test])[0, 1]
        pred = model.predict([X_test])[0]
        risk = "HIGH" if proba > 0.66 else "MEDIUM" if proba > 0.33 else "LOW"
        print(
            f"  Year={X_test[0]}, Month={X_test[1]}: proba={proba:.4f}, pred={pred}, risk={risk}")

    # Save model
    output_path = "models/simple_logistic_model.joblib"
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")

    # Also save feature names
    import json
    with open("models/feature_names_fixed.json", "w") as f:
        json.dump(["Year_mean", "Month_mean"], f)

    return output_path


if __name__ == "__main__":
    print("Model Compatibility Fix")
    print("=======================")

    # Try to fix existing models
    fix_model_compatibility()

    # Create new simple model
    new_model_path = create_simple_model()

    print(f"\nTo use the new model, update docker-compose with:")
    print(f"  MODEL_LOCAL_PATH: /app/{new_model_path}")
    print(f"  MODEL_FEATURES_PATH: /app/models/feature_names_fixed.json")
