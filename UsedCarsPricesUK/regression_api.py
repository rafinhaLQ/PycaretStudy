# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("regression_api")

# Create input/output pydantic models
input_model = create_model("regression_api_input", **{'Mileage(miles)': 27500, 'Registration Year': 2016, 'Previous Owners': 2.0, 'Fuel Type': 1, 'Body Type': 'Hatchback', 'Engine': 1.0, 'Gearbox': 1, 'Doors': 3.0, 'Seats': 5.0, 'Emission Class': 'Euro 6', 'Service History': 0})
output_model = create_model("regression_api_output", prediction=6900)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
