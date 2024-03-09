from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
import pickle
try:
    from ..prediction.nn_models import LSTM_TimeSeriesModel_2
except ImportError:
    from nn_models import LSTM_TimeSeriesModel_2

app = FastAPI()



# Load the PyTorch model checkpoint
model_checkpoint_path = 'model_checkpoint.pth'


model = LSTM_TimeSeriesModel_2(2,2,2)
model.load_state_dict(torch.load(model_checkpoint_path,map_location=torch.device('cpu'))['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Define request body schema
class Item(BaseModel):
    input_data: List[List[float]]

# Define FastAPI route
@app.post("/predict")
async def get_prediction(item: Item):

    input_data = torch.as_tensor(item.input_data, dtype=torch.float32).view(1, 24, 2)

    with torch.no_grad():
        output = model(input_data)

    return {"prediction": output.tolist()}  # Convert tensor to list for JSON serialization
