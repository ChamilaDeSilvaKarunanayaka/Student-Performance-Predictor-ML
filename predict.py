import joblib
import pandas as pd

# Load model
model = joblib.load("models/student_model.pkl")

# User input
hours = int(input("Enter study hours: "))
attendance = int(input("Enter attendance: "))

# Convert to DataFrame (no warning)
data = pd.DataFrame([[hours, attendance]], columns=["hours", "attendance"])

# Prediction
prediction = model.predict(data)

# Output
if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")