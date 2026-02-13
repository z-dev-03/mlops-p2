import pickle
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

print("=== ML CI Pipeline ===")

# Training data
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# Create and train model
model = LinearRegression()
model.fit(X, y)

print("Model trained")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved")

# Validation test
prediction = model.predict([[4]])[0]
print(f"Prediction: {prediction:.1f} (Expected: 8.0)")

# CI validation check
if abs(prediction - 8.0) < 0.1:
    print("VALIDATION PASSED")
    sys.exit(0)  # Success
else:
    print("VALIDATION FAILED")
    sys.exit(1)  # Failure
