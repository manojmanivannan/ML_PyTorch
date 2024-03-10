from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
from typing import List
import pickle
try:
    from ..prediction.nn_models import LSTM_TimeSeriesModel_2
except ImportError:
    from nn_models import LSTM_TimeSeriesModel_2

app = FastAPI()

# Load mean and std
with open('overall_mean.pkl', 'rb') as filehandler:
    overall_mean = pickle.load(filehandler)
    
with open('overall_std.pkl', 'rb') as filehandler:
    overall_std = pickle.load(filehandler)

# Load the PyTorch model checkpoint
model_checkpoint_path = 'model_checkpoint.pth'


model = LSTM_TimeSeriesModel_2(2,2,2)
model.load_state_dict(torch.load(model_checkpoint_path,map_location=torch.device('cpu'))['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Define request body schema
class Item(BaseModel):
    input_data: List[List[float]]
    mean: List[float]
    std: List[float]

# Define FastAPI route
@app.post("/predict")
async def get_prediction(item: Item):

    normalized_input = (item.input_data - np.array(item.mean)) / np.array(item.std)
    # print(normalized_input)
    input_data = torch.as_tensor(normalized_input, dtype=torch.float32).view(1, 24, 2)

    with torch.no_grad():
        output = model(input_data)
    
    output = (output * np.array(item.std)) + np.array(item.mean)
    result = {"prediction": output.tolist()}

    return result  # Convert tensor to list for JSON serialization
