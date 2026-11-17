import copy
import math

import torch
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


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, images):
        x = F.silu(self.conv1(images))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        x = self.flatten(x)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        self.register_buffer(
            "angular_speeds",
            2 * math.pi * frequencies.view(1, 1, 1, -1),
        )

    def forward(self, noise_variances):
        angles = self.angular_speeds * noise_variances
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.conv1(x))
        return self.conv2(x) + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels

        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)

        noise_embedding = self.embedding(noise_variances)
        noise_embedding = F.interpolate(
            noise_embedding.permute(0, 3, 1, 2),
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        x = torch.cat([x, noise_embedding], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


def offset_cosine_diffusion_schedule(
    diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95
):
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(angles), torch.cos(angles)


class DiffusionModel(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.ema_network = copy.deepcopy(network)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.register_buffer("normalizer_mean", torch.tensor(0.5))
        self.register_buffer("normalizer_std", torch.tensor(0.5))

    def denormalize(self, images):
        return (
            images * self.normalizer_std + self.normalizer_mean
        ).clamp(0, 1)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()

        predicted_noise = network(noisy_images, noise_rates.square())
        predicted_images = (
            noisy_images - noise_rates * predicted_noise
        ) / signal_rates
        return predicted_noise, predicted_images

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noise = torch.randn_like(images)
        times = torch.rand(images.size(0), 1, 1, 1, device=images.device)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(times)
        noisy_images = signal_rates * images + noise_rates * noise

        optimizer.zero_grad(set_to_none=True)
        predicted_noise, _ = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True
        )
        loss = loss_fn(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for averaged, current in zip(
                self.ema_network.parameters(), self.network.parameters()
            ):
                averaged.mul_(self.ema_decay).add_(
                    current, alpha=1 - self.ema_decay
                )
            for averaged, current in zip(
                self.ema_network.buffers(), self.network.buffers()
            ):
                averaged.copy_(current)

        return loss.item()

    @torch.no_grad()
    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noise = torch.randn_like(images)
        times = torch.rand(images.size(0), 1, 1, 1, device=images.device)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(times)
        noisy_images = signal_rates * images + noise_rates * noise
        predicted_noise, _ = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        return loss_fn(predicted_noise, noise).item()

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive.")

        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            times = torch.full(
                (initial_noise.size(0), 1, 1, 1),
                1 - step * step_size,
                device=initial_noise.device,
                dtype=initial_noise.dtype,
            )
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(times)
            predicted_noise, predicted_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_noise_rates, next_signal_rates = (
                offset_cosine_diffusion_schedule((times - step_size).clamp(0, 1))
            )
            current_images = (
                next_signal_rates * predicted_images
                + next_noise_rates * predicted_noise
            )
        return predicted_images

    @torch.no_grad()
    def generate(self, num_images=1, diffusion_steps=100):
        if num_images < 1:
            raise ValueError("num_images must be positive.")

        parameter = next(self.network.parameters())
        initial_noise = torch.randn(
            num_images,
            self.network.num_channels,
            self.network.image_size,
            self.network.image_size,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.denormalize(
            self.reverse_diffusion(initial_noise, diffusion_steps)
        )


def get_model(model_name):
    if model_name == 'CNN':
        return CNN()
    if model_name == 'GAN':
        return GAN()
    if model_name == 'Energy':
        return EnergyModel()
    if model_name == 'Diffusion':
        return DiffusionModel(UNet())
    raise ValueError(
        f'Unsupported model: {model_name!r}. '
        'Choose "CNN", "GAN", "Energy", or "Diffusion".'
    )
