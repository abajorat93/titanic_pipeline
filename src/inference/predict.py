import os
from typing import List, Tuple
import joblib
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from fastapi import FastAPI, Depends
import pandas as pd
from src.config import config


class PredictionInput(BaseModel):
    pclass: int
    name: str
    sex: str
    age: str
    sibsp: int
    parch: int
    ticket: str
    fare: str
    cabin: str
    embarked: str
    boat: str
    body: str
    home_dest: str

class PredictionOutput(BaseModel):
    prediction: int
    
    
class TitanicModel:
    model: Pipeline
    def load_model(self):
        """Loads the model"""
        model_file = config.PRODUCTION_MODEL
        self.model = joblib.load(model_file)
        
    def predict(self, input: PredictionInput) -> PredictionOutput:
        """Runs a prediction"""
        if not self.model:
            raise RuntimeError("Model is not loaded")
        print(f"Raw Input: {input.dict()}")
        df = pd.DataFrame([input.dict()])
        prediction = self.model.predict(df)
        print(f"Prediction: {prediction}")
        return PredictionOutput(prediction=prediction)

app = FastAPI()
titanic_model = TitanicModel()

@app.post("/prediction")
async def prediction(
    output: PredictionOutput = Depends(titanic_model.predict)) -> PredictionOutput:
    return output

@app.on_event("startup")
async def startup():
    titanic_model.load_model()