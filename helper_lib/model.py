import torch.nn as nn
import torch.nn.functional as F

from .generator import Generator

class CNN(nn.Module):

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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.leaky_relu(self.conv1(images))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        return self.sigmoid(self.fc(x))


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.generator = Generator(z_dim)
        self.discriminator = Discriminator()

    def forward(self, noise):
        return self.generator(noise)


def get_model(model_name):
    if model_name == 'CNN':
        return CNN()
    if model_name == 'GAN':
        return GAN()
    raise ValueError(f'Unsupported model: {model_name!r}. Choose "CNN" or "GAN".')
