import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size, dropout_rate=0.6):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flattened_size = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(self.flattened_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
    

    # Gets the size of the first linear layer
    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.numel()


    def forward(self, x):
        # First CNN layer
        x = self.pool1(F.relu(self.conv1(x)))

        # Second CNN layer
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor

        # First FCN layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second FCN layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Third FCN layer
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.labels:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]