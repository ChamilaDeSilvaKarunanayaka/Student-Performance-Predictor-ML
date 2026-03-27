import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input (X) and Output (y)
x = data[["hours", "attendance"]]
y = data["result"]

# Encode labels (fail=0, pass=1)
y = y.map({"fail": 0, "pass": 1})

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)
joblib.dump(model, "models/student_model.pkl") # Save model

# Predictions (fixed with DataFrame → no warning)
p1 = model.predict(pd.DataFrame([[5, 75]], columns=["hours", "attendance"]))
p2 = model.predict(pd.DataFrame([[2, 40]], columns=["hours", "attendance"]))
p3 = model.predict(pd.DataFrame([[8, 95]], columns=["hours", "attendance"]))

# Print results
print("5h,75%:", "PASS" if p1[0] == 1 else "FAIL")
print("2h,40%:", "PASS" if p2[0] == 1 else "FAIL")
print("8h,95%:", "PASS" if p3[0] == 1 else "FAIL")

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Hours vs Result
plt.scatter(data["hours"], y, c=y)
plt.xlabel("Study Hours")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Study Hours vs Result")
plt.show()

# Attendance vs Result
plt.scatter(data["attendance"], y, c=y)
plt.xlabel("Attendance")
plt.ylabel("Result (0 = Fail, 1 = Pass)")
plt.title("Attendance vs Result")
plt.show()