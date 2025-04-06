import torch
import torch.nn as nn

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ClassificationModel, self).__init__()
        self.linear_sequential_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size[2], output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.linear_sequential_stack(x)
        return logits

class DetectorModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DetectorModel, self).__init__()
        self.linear_sequential_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_size[2], output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_sequential_stack(x)
        return logits