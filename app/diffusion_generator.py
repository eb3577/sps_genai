from io import BytesIO
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from helper_lib.generator import generate_diffusion_samples
from helper_lib.model import get_model


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "diffusion"
    / "diffusion.pth"
)


def generate_diffusion_image():
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "No trained diffusion model is available. Run train_diffusion.py first."
        )

    try:
        model = get_model("Diffusion")
        weights = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights)
    except Exception as exc:
        raise RuntimeError(
            "The diffusion checkpoint could not be loaded. "
            "Try again after training finishes."
        ) from exc

    model.eval()
    samples = generate_diffusion_samples(
        model, device="cpu", num_samples=1, diffusion_steps=100
    )
    image = to_pil_image(samples[0])

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
