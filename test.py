import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer
import os
from typing import Dict, List


def main():
    NUM_EPOCHS = 7
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count() - 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformation for images
    train_data_transform = transforms.Compose(
        [
            # Resize the images to 64x64
            transforms.Resize(size=(64, 64)),
            transforms.TrivialAugmentWide(num_magnitude_bins=31),
            transforms.ToTensor(),
        ]
    )
    test_data_transform = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
        ]
    )

    data_path = Path("data/")
    image_path = data_path / "Tumors"
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    # image_path_list = list(image_path.glob("*/*/*.jpg"))
    # plot_transformed_images(image_path_list, transform=data_transform, n=3)

    train_data = datasets.ImageFolder(
        root=train_dir,  # target folder of images
        transform=train_data_transform,
    )
    print(train_data.classes)


if __name__ == "__main__":
    main()
