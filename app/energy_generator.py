from io import BytesIO
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from helper_lib.generator import generate_energy_samples
from helper_lib.model import get_model


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "energy"
    / "energy.pth"
)


def generate_energy_image():
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "No trained energy model is available. Run train_energy.py first."
        )

    try:
        model = get_model("Energy")
        weights = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights)
    except Exception as exc:
        raise RuntimeError(
            "The energy checkpoint could not be loaded. "
            "Try again after training finishes."
        ) from exc

    model.eval()
    initial_images = torch.rand(1, 3, 32, 32) * 2 - 1
    samples = generate_energy_samples(
        model, initial_images, steps=256, step_size=10.0, noise_std=0.005
    )
    image = to_pil_image(((samples[0] + 1) / 2).clamp(0, 1))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
