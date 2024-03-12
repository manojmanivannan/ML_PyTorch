from ts.torch_handler.base_handler import BaseHandler
import json
import numpy as np


class ONNXHandler(BaseHandler):
    def __init__(self):
        super().__init__()


    def initialize(self, context):
        super().initialize(context)
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        with open(model_dir + "/model_data.json", "r") as f:
            model_data = json.loads(f.read())
            self.window_size = model_data["window_size"]
            self.selected_cols = model_data["selected_columns"]
            self.original_data_mean = model_data["overall_mean"]
            self.original_data_std = model_data["overall_std"]
            
            
            
    def preprocess(self,input_data):
        
        reshaped_data =np.array(input_data, dtype=np.float32).reshape(1, self.window_size, len(self.selected_cols))
        normalized_data = (reshaped_data - np.array(self.data_mean)) / np.array(self.data_std)
        return normalized_data
        
    def handle(self,data, context):
        body = json.loads(data[0]['body'])
        input_data = body['input']
        self.data_mean = body['mean']
        self.data_std = body['std'] 

        
        if hasattr(self.model, "run"):
            
            normalized_input = self.preprocess(input_data)
            
            results = self.model.run(None, {"input": normalized_input.tolist()})
            
            return self.postprocess(results[0])
        
    def postprocess(self, data):
        print('Prediction',data)
        denomalized_results = np.array(data * np.array(self.data_std)) + np.array(self.data_mean)
        return  denomalized_results.tolist()