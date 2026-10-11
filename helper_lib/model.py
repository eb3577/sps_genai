import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Assignment 2 architecture for 64x64 RGB images and 10 classes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16 channels, 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32 channels, 16x16
        x = x.flatten(start_dim=1)  # 8,192 features per image
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Raw scores for CrossEntropyLoss.


def get_model(model_name):
    if model_name == 'CNN':
        return CNN()
    raise ValueError(f'Unsupported model: {model_name!r}. Choose "CNN".')
