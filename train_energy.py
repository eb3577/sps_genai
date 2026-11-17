from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.data_loader import get_cifar10_generative_loader
from helper_lib.generator import generate_energy_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_energy


BASE_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training energy model on {device}")

    loader = get_cifar10_generative_loader(
        BASE_DIR / "data", batch_size=128, normalize=True
    )
    model = get_model("Energy").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model = train_energy(
        model, loader, optimizer, device=device, epochs=10,
        alpha=0.1, steps=60, step_size=10.0, noise_std=0.005,
    )

    output_dir = BASE_DIR / "checkpoints" / "energy"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "energy.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Energy model weights saved: {weights_path}")

    initial_images = torch.rand(16, 3, 32, 32, device=device) * 2 - 1
    samples = generate_energy_samples(
        model, initial_images, steps=256, step_size=10.0, noise_std=0.005
    )
    samples_path = output_dir / "samples.png"
    save_image(((samples + 1) / 2).cpu(), samples_path, nrow=4)
    print(f"Sample grid saved: {samples_path}")


if __name__ == "__main__":
    main()
