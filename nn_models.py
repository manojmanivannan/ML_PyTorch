

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y,label,window_size,output_size):
        self.X = torch.tensor(X, dtype=torch.float32).view(-1, window_size, output_size)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, output_size)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"{label} tensors moved to GPU")
            self.X = self.X.to(device)
            self.y = self.y.to(device)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTM_TimeSeriesModel_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_TimeSeriesModel_1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Additional fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Additional fully connected layer

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)  # Applying ReLU activation
        out = self.fc2(out)
        return out
    


class LSTM_TimeSeriesModel_2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_TimeSeriesModel_2, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerTimeSeriesModel, self).__init__()
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.linear_out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # Expected input dimension (L, N, E) - Length, Batch size, Embedding
        x = x.permute(1, 0, 2)  # permute to fit transformer's expected input shape
        x = self.transformer(x, x)
        x = self.linear_out(x)
        x = x.permute(1, 0, 2)  # permute back to original shape
        return x[:, -1, :]  # return only the last prediction for each sequence
