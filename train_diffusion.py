from pathlib import Path

import torch
from torchvision.utils import save_image

from helper_lib.data_loader import get_cifar10_generative_loader
from helper_lib.generator import generate_diffusion_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion


BASE_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training diffusion model on {device}")

    loader = get_cifar10_generative_loader(
        BASE_DIR / "data", batch_size=64, normalize=False
    )
    model = get_model("Diffusion").to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.0001)

    model = train_diffusion(
        model, loader, criterion, optimizer, device=device, epochs=10
    )

    output_dir = BASE_DIR / "checkpoints" / "diffusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "diffusion.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Diffusion model weights saved: {weights_path}")

    samples = generate_diffusion_samples(
        model, device, num_samples=16, diffusion_steps=100
    )
    samples_path = output_dir / "samples.png"
    save_image(samples, samples_path, nrow=4)
    print(f"Sample grid saved: {samples_path}")


if __name__ == "__main__":
    main()
