import json
from pathlib import Path
from pyexpat import model

import torch
import torchvision
from torchvision import transforms
from torch import nn

from going_modular import data_setup, engine, helper_functions, utils


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 300
    # torch.set_default_device(DEVICE)
    data = "Data/"

    # Setup directory paths to train and test images
    data_type = "Tumors"
    train_dir = data + data_type + "/train"
    test_dir = data + data_type + "/test"
    # train_dir = data + "Tumors" + "/train"
    # test_dir = data + "Tumors" + "/test"
    vit, vit_transforms, vit_transforms_augmented = create_vit_model(num_classes=4, seed=43)
    (
        train_dataloader_vit,
        test_dataloader_vit,
        class_names,
    ) = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=vit_transforms_augmented,
        test_transform=vit_transforms,
        batch_size=128,
    )

    # Setup optimizer
    optimizer = torch.optim.Adam(params=vit.parameters(), lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Setup the PyTorch TensorBoard logger
    writer = helper_functions.create_writer(
        experiment_name="Early stopping",
        model_name="vit",
        extra=f"{EPOCHS}_epochs",
    )

    # Set seeds for reproducibility and train the model
    helper_functions.set_seeds()
    vit_results = engine.train(
        model=vit.to(DEVICE),
        train_dataloader=train_dataloader_vit,
        test_dataloader=test_dataloader_vit,
        epochs=EPOCHS,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        writer=writer,
        patience=17,   
    )
    # Save model
    utils.save_model(model=vit, model_name="vit" + ".pth", target_dir="models")
    # Count number of parameters in ViT
    vit_total_params = sum(torch.numel(param) for param in vit.parameters())
    # Get the model size in bytes then convert to megabytes
    pretrained_vit_model_size = Path("models/vit.pth").stat().st_size // (
        1024 * 1024
    )  # division converts bytes to megabytes (roughly)
    print(f"Pretrained VIT feature extractor model size: {pretrained_vit_model_size} MB")
    # Create a dictionary with VIT statistics
    vit_stats = {
        "test_loss": vit_results["test_loss"][-1],
        "test_acc": vit_results["test_acc"][-1],
        "number_of_parameters": vit_total_params,
        "model_size (MB)": pretrained_vit_model_size,
    }
    # Save the dictionary as a JSON file
    with open("vit_stats.json", "w") as json_file:
        json.dump(vit_stats, json_file)

    helper_functions.plot_loss_curves(vit_results)


def create_vit_model(num_classes: int, seed: int = 43):
    """Creates a ViT-B/16 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of target classes. Defaults to 3.
        seed (int, optional): random seed value for output layer. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT-B/16 feature extractor model.
        transforms (torchvision.transforms): ViT-B/16 image transforms.
    """
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    vit_transforms = weights.transforms()
    vit_transforms_augmented = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        vit_transforms
    ])
    model = torchvision.models.vit_b_16(weights=weights)

    # Freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last block of the transformer encoder
    for param in model.encoder.layers[-1].parameters():
        param.requires_grad = True

    # Unfreeze the final layer normalization layer (often helps)
    for param in model.encoder.ln.parameters():
        param.requires_grad = True

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(seed)
    model.heads = nn.Sequential(
        nn.Linear(
            in_features=768,  # keep this the same as original model
            out_features=num_classes,
        )
    )  # update to reflect target number of classes

    return model, vit_transforms, vit_transforms_augmented


if __name__ == "__main__":
    main()
