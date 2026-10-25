import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(noise.size(0), 128, 7, 7)
        x = self.relu(self.bn1(self.deconv1(x)))
        return self.tanh(self.deconv2(x))


def generate_samples(model, device, num_samples=10):
    """Accept a Generator and return display-ready digit images."""
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            noise = torch.randn(num_samples, model.z_dim, device=device)
            images = model(noise)
            return ((images + 1) / 2).cpu()
    finally:
        model.train(was_training)
