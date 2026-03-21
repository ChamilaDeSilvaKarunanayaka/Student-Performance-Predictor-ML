import pandas as pd
from sklearn.model_selection import train_test_split   # Importing the function to split the dataset into training and testing sets
from sklearn.linear_model import LogisticRegression    # Importing the Logistic Regression model
from sklearn.metrics import accuracy_score          # Importing the function to evaluate the accuracy of the model

data = pd.read_csv("data/student_data.csv") 

x = data[[ "hours", "attendance" ]] 
y = data["result"]

y = y.map({"fail": 0, "pass": 1}) #Encoding the labels

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
) #Splitting the dataset into training and testing sets, with 20% of the data reserved for testing

model = LogisticRegression() #Creating an instance of the Logistic Regression model

model.fit(X_train, y_train) #Training the model using the training data

prediction = model.predict([[5, 75]]) #Making a prediction for a student who studies 5 hours and has 75% attendance)
prediction = model.predict([[2, 40]])
prediction = model.predict([[8, 95]])

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)