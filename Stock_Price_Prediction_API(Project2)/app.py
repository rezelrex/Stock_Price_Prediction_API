# -------------------------------------------------------------------------
# Step 1: Install Dependencies for the API
# -------------------------------------------------------------------------
# Before running this script, you need to install FastAPI and a web server.
# Open your terminal and run:
# pip install fastapi uvicorn[standard] torch
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Check if a GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------
# Step 2: Define the same LSTM Model class
# -------------------------------------------------------------------------
# The model class definition must be the same as the one used for training.
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -------------------------------------------------------------------------
# Step 3: Initialize the Model and Load the Trained Weights
# -------------------------------------------------------------------------
# IMPORTANT: In a real project, you would save your trained model from the
# previous script and load it here. For this example, we'll create a dummy model.
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.eval() # Set the model to evaluation mode

# -------------------------------------------------------------------------
# Step 4: Define the API
# -------------------------------------------------------------------------
# This creates the FastAPI application instance.
app = FastAPI(
    title="Financial Time-Series Prediction API",
    description="A simple API to predict stock prices using a trained LSTM model."
)

# -------------------------------------------------------------------------
# Step 5: Define the Input Data Model
# -------------------------------------------------------------------------
# We use Pydantic to validate the incoming request body. This ensures
# that the data sent to the API is in the correct format.
class PredictionInput(BaseModel):
    sequence: List[float]

# -------------------------------------------------------------------------
# Step 6: Create the API Endpoint
# -------------------------------------------------------------------------
# This is the endpoint that will receive the data and return a prediction.
@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Predicts the next stock price based on a sequence of historical prices.
    
    Args:
        data (PredictionInput): A JSON object containing a list of 60 normalized prices.
    
    Returns:
        dict: A JSON object with the predicted price.
    """
    try:
        # Convert the input list to a PyTorch tensor and reshape it
        input_tensor = torch.tensor(data.sequence).float().view(1, -1, 1).to(device)
        
        with torch.no_grad(): # Disable gradient calculations for inference
            prediction = model(input_tensor)
        
        # In a real scenario, you would inverse-transform the prediction
        # prediction_scaled = scaler.inverse_transform(prediction.cpu().numpy())
        
        return {"prediction": prediction.item()}
        
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------------------
# Step 7: How to run this API
# -------------------------------------------------------------------------
# To run this API from your terminal, navigate to the directory where this
# file is saved and run the following command:
# uvicorn app:app --reload
#
# The API will be available at http://127.0.0.1:8000
# You can test the /predict endpoint using a tool like Postman or a simple
# Python script.
