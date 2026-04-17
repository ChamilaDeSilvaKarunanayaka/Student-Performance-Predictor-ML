from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (frontend connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# Input structure
class StudentData(BaseModel):
    hours: float
    attendance: float

# Test route
@app.get("/")
def home():
    return {"message": "API running"}

# Predict route
@app.post("/predict")
def predict(data: StudentData):
    input_data = [[data.hours, data.attendance]]
    
    prediction = model.predict(input_data)

    result = "PASS" if prediction[0] == 1 else "FAIL"

    return {"result": result}