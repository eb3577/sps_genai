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


def generate_diffusion_samples(
    model, device, num_samples=10, diffusion_steps=100
):
    """Return CPU images in [0, 1] using reverse diffusion."""
    if num_samples < 1 or diffusion_steps < 1:
        raise ValueError("num_samples and diffusion_steps must be positive.")

    model.to(device)
    training_modes = {
        module: module.training for module in model.modules()
    }
    model.eval()

    try:
        with torch.no_grad():
            return model.generate(
                num_images=num_samples,
                diffusion_steps=diffusion_steps,
            ).cpu()
    finally:
        for module, training in training_modes.items():
            module.training = training


def generate_energy_samples(model, images, steps, step_size, noise_std):
    """Refine initial images using Langevin sampling; return values in [-1, 1]."""
    if steps < 1 or step_size <= 0 or noise_std < 0:
        raise ValueError(
            "steps and step_size must be positive; noise_std cannot be negative."
        )

    was_training = model.training
    model.eval()
    images = images.detach().clone()

    try:
        for _ in range(steps):
            with torch.no_grad():
                noise = torch.randn_like(images) * noise_std
                images = (images + noise).clamp(-1, 1)

            with torch.enable_grad():
                images.requires_grad_(True)
                energy = model(images)
                gradients, = torch.autograd.grad(energy.sum(), images)

            with torch.no_grad():
                gradients = gradients.clamp(-0.03, 0.03)
                images = (images - step_size * gradients).clamp(-1, 1)

        return images.detach()
    finally:
        model.train(was_training)


class EnergyReplayBuffer:
    def __init__(self, model, device, max_size=8192):
        if max_size < 1:
            raise ValueError("max_size must be positive.")

        self.model = model
        self.device = device
        self.max_size = max_size
        self.examples = [
            torch.rand(1, 3, 32, 32, device=device) * 2 - 1
            for _ in range(min(128, max_size))
        ]

    def sample(self, batch_size, steps, step_size, noise_std):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        n_new = int((torch.rand(batch_size) < 0.05).sum().item())
        parts = []

        if n_new:
            parts.append(
                torch.rand(n_new, 3, 32, 32, device=self.device) * 2 - 1
            )

        n_old = batch_size - n_new
        if n_old:
            indices = torch.randint(len(self.examples), (n_old,)).tolist()
            parts.append(torch.cat([self.examples[i] for i in indices]))

        images = torch.cat(parts)
        samples = generate_energy_samples(
            self.model, images, steps, step_size, noise_std
        )

        # Store independent copies without retaining a gradient graph.
        new_examples = [
            image.detach().clone()
            for image in samples[:self.max_size].split(1)
        ]
        self.examples = (new_examples + self.examples)[:self.max_size]
        return samples
