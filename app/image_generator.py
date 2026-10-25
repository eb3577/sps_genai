from io import BytesIO
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from helper_lib.generator import Generator, generate_samples


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "assignment3"
    / "generator.pth"
)


def generate_digit():
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            "No trained generator is available. Run train_gan.py first."
        )

    try:
        model = Generator()
        weights = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights)
    except Exception as exc:
        raise RuntimeError(
            "The generator checkpoint could not be loaded. "
            "Try again after training finishes."
        ) from exc

    model.eval()
    samples = generate_samples(model, device="cpu", num_samples=1)
    image = to_pil_image(samples[0])

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
