import joblib

from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()
model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

# Define a Pydantic model for input data validation

class InputFeatures(BaseModel):
    current_value : float
    goals: float
    


@app.post("/predict")
def predict_cluster(input_data: InputFeatures):
    # Preprocess input
    scaled_data = scaler.transform([[input_data.current_value, input_data.goals]])
    
    # Predict cluster
    cluster = model.predict(scaled_data)[0]
    
    return {"cluster": int(cluster)}

