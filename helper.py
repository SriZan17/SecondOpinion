import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_two_models(model_0_df, model_1_df):
    # Setup a plot
    plt.figure(figsize=(15, 10))

    # Get number of epochs
    epochs = range(len(model_0_df))

    # Plot train loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot test loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
    plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
    plt.title("Test Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot train accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
    plt.title("Train Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot test accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
    plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator.
        Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
        fig.savefig(f"{image_path.stem}_transformed.png")


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names,
    transform=None,
    device: torch.device = device,
):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
