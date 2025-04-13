import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.6):
        super(ClassificationModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        # First layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Third layer
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        return x