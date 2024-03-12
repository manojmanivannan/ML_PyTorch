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
            
    def handle(self,data, context):
        body = (data[0]['body'])
        
        if hasattr(self.model, "run"):
            data = np.array(body, dtype=np.float32).reshape(1, self.window_size, len(self.selected_cols))
            
            results = self.model.run(None, {"input": data})
            return [i for i in results[0]]