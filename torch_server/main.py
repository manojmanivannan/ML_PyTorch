from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from typing import List
import json
import onnxruntime
import numpy as np

app = FastAPI()

# Load the ONNX model
session = onnxruntime.InferenceSession("model_onnx.onnx")

# Load model metadata
with open('model_data.json', 'r') as f:
    model_metadata = json.load(f)


# Define a function to perform inference
def inference(input_data):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    input_data = np.array(input_data).reshape(1, model_metadata['window_size'],len(model_metadata['selected_columns']))

    pred = session.run([output_name], {input_name: input_data.astype(np.float32)})[0]
    return pred[-1]


# Define request body schema
class Item(BaseModel):
    input_data: List[List[float]]
    mean: List[float]
    std: List[float]

# Define FastAPI route
@app.post("/predict")
async def get_prediction(item: Item):

    normalized_input = (item.input_data - np.array(item.mean)) / np.array(item.std)

    output = inference(normalized_input)
    
    output = np.array(output * np.array(item.std)) + np.array(item.mean)

    result = {"prediction": output.tolist()}

    return result
