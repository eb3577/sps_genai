from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.data_loader import get_mnist_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan


BASE_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training GAN on {device}")

    loader = get_mnist_data_loader(BASE_DIR / "data", batch_size=64)
    model = get_model("GAN")
    model = train_gan(model, loader, device=device, epochs=10)

    output_dir = BASE_DIR / "checkpoints" / "assignment3"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "generator.pth"
    torch.save(model.generator.state_dict(), weights_path)

    samples = generate_samples(model.generator, device, num_samples=16)
    samples_path = output_dir / "samples.png"
    save_image(samples, samples_path, nrow=4)

    print(f"Generator weights saved: {weights_path}")
    print(f"Sample grid saved: {samples_path}")


if __name__ == "__main__":
    main()
