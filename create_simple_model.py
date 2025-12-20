import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import json

print('Creating simple working model...')

# Create and train a simple model
np.random.seed(42)
X = np.column_stack([
    np.random.uniform(2018, 2020, 1000),
    np.random.uniform(1, 12, 1000)
])
y = ((X[:, 0] > 2018.5) & (X[:, 1] > 6)).astype(int)

model = LogisticRegression(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/simple_working_model.joblib')

# Save feature names
with open('models/feature_names_simple.json', 'w') as f:
    json.dump(['Year_mean', 'Month_mean'], f)

print('Created simple_working_model.joblib')
