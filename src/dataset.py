import os
import torch
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32):
    """
    Load training and validation datasets with augmentation.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader
