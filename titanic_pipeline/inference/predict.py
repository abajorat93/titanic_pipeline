import os
from typing import List, Tuple
import joblib
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from fastapi import FastAPI, Depends, BackgroundTasks
import pandas as pd
from titanic_pipeline.config import config


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

class PredictionOutput(BaseModel):
    prediction: int
    
    
class TitanicModel:
    staging_model: Pipeline
    prod_model: Pipeline
    def load_model(self):
        """Loads the model"""
        self.prod_model = joblib.load(config.PRODUCTION_MODEL_FILE)
        self.staging_model = joblib.load(config.STAGING_MODEL_FILE)
        
    def staging_predict(self, input: PredictionInput):
        df = pd.DataFrame([input.dict()])
        if not self.staging_model:
            raise RuntimeError("Model is not loaded")
        prediction = self.staging_model.predict(df)
        print(f"Prediction: {prediction}")

    
    def predict(self, input: PredictionInput, background_tasks: BackgroundTasks) -> PredictionOutput:
        """Runs a prediction"""        
        print(f"Raw Input: {input.dict()}")
        df = pd.DataFrame([input.dict()])
        
        if not self.prod_model:
            raise RuntimeError("Model is not loaded")
        prediction = self.prod_model.predict(df)
        background_tasks.add_task(self.staging_predict, input)
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