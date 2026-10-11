import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        # CIFAR-10 images are 32x32; the assignment requires 64x64 inputs.
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )
    return loader
