# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("ASCVD_Risk")

# Create input/output pydantic models
input_model = create_model("ASCVD_Risk_input", **{'isMale': 1, 'isBlack': 1, 'isSmoker': 0, 'isDiabetic': 0, 'isHypertensive': 1, 'Age': 68, 'Systolic': 99, 'Cholesterol': 166, 'HDL': 61})
output_model = create_model("ASCVD_Risk_output", prediction=11.1)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
