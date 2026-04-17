import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Get base directory (project root)
base_dir = os.path.dirname(os.path.dirname(__file__))

# Load dataset
data_path = os.path.join(base_dir, "data", "student_data.csv")
data = pd.read_csv(data_path)

# Features (X) and Labels (y)
X = data[["hours", "attendance"]]
y = data["result"].map({"fail": 0, "pass": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Accuracy (optional)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model to backend folder
model_path = os.path.join(base_dir, "backend", "model.pkl")
joblib.dump(model, model_path)

print("Model trained and saved to backend/model.pkl")