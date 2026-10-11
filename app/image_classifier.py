from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from helper_lib.model import get_model


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints' / 'assignment2' / 'best'
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def classify_image(image_bytes):
    # Match training: RGB, 64x64 pixels, tensor values between 0 and 1.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = transforms.Resize((64, 64))(image.convert('RGB'))
            inputs = transforms.ToTensor()(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError('Upload a valid image, such as a PNG or JPEG.') from exc

    # The trainer saves a file in best/ whenever validation accuracy improves.
    checkpoints = list(CHECKPOINT_DIR.glob('model_epoch_*.pth'))
    if not checkpoints:
        raise RuntimeError('No trained checkpoint is available yet.')
    checkpoint_path = max(
        checkpoints, key=lambda path: int(path.stem.rsplit('_', 1)[1])
    )
    try:
        model = get_model('CNN')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as exc:
        # A training process may still be writing the checkpoint.
        raise RuntimeError('The model checkpoint could not be loaded. Try again after training finishes.') from exc

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        class_id = outputs.argmax(dim=1).item()
    return {'class': CLASSES[class_id], 'class_id': class_id}
