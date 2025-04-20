import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size, dropout_rate=0.6):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flattened_size = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(self.flattened_size, hidden_sizes[0])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn4 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
    

    # Gets the size of the first linear layer
    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            return x.numel()


    def forward(self, x):
        # First CNN layer
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second CNN layer
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor

        # First FCN layer
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second FCN layer
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data_dir: str, annotation_file: str, encoder, transform=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir
        self.annotations = pd.read_csv(data_dir + "/imgs/" + annotation_file)
        self.encoder = encoder
        self.transform = transform

        self.img_to_tensor = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        imgs = self.img_to_tensor(Image.open(img_id)).to(self.device)

        labels = self.annotations.iloc[index, 1]
        labels = np.array(labels).reshape(-1, 1)
        try:
            labels = self.encoder.transform(labels)
        except TypeError:
            print("Error!")
            print(index)
            print(self.annotations.iloc[index])
            print(labels)
            exit(0)
        labels = torch.tensor(labels).to(self.device)
        labels = labels.squeeze().to(torch.float32)

        imgs = imgs.view(-1, *imgs.shape[1:])
        imgs = imgs.to(torch.float32)
        

        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, labels

    def get_data_shape(self):
        return tuple(self.__getitem__(0)[0].shape[1:])
        

    def get_label_shape(self):
        return self.encoder.transform(np.array(self.annotations.iloc[0, 1]).reshape(-1, 1)).shape[1]