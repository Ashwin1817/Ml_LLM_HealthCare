from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re

app = FastAPI()

# Load model & encoder
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load dataset to get column order
df = pd.read_csv("Training.csv")
symptom_columns = df.drop("prognosis", axis=1).columns


class SymptomRequest(BaseModel):
    text: str


def extract_symptoms_from_text(user_text, symptom_columns):
    user_text = user_text.lower()
    detected = []

    for symptom in symptom_columns:
        clean_symptom = symptom.replace("_", " ")
        if re.search(r"\b" + re.escape(clean_symptom) + r"\b", user_text):
            detected.append(symptom)

    return detected


@app.post("/predict")
def predict_from_text(data: SymptomRequest):

    symptoms = extract_symptoms_from_text(data.text, symptom_columns)

    input_data = np.zeros(len(symptom_columns))

    for symptom in symptoms:
        index = list(symptom_columns).index(symptom)
        input_data[index] = 1

    input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    disease = le.inverse_transform(prediction)[0]
    confidence = round(float(max(probability[0])) * 100, 2)

    return {
        "predicted_disease": disease,
        "confidence": confidence,
        "detected_symptoms": symptoms
    }