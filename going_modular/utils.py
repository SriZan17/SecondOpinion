"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from torchmetrics import ConfusionMatrix
from tqdm.auto import tqdm
import os
from torchvision import datasets
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc(y_true, y_score, class_names, save_path):
    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))  # shape (N, C)

    # Compute per-class ROC
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    # Aggregate all FPRs
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Interpolate
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average ROC (AUC = {roc_auc['micro']:.3f})",
             color="deeppink", linestyle=":", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"macro-average ROC (AUC = {roc_auc['macro']:.3f})",
             color="navy", linestyle=":", linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, c in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=c,
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})", linewidth=1.5)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (One-vs-Rest)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def calculate_confusion_matrix(
    class_names: list,
    model,  # y_true: torch.Tensor, y_pred: torch.Tensor
    test_dir: str,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    transform=None,
):
    """Calculates a PyTorch confusion matrix using the ConfusionMatrix class from torchmetrics.

    Args:
      class_names: A list of class names in the order of the confusion matrix.
      y_true: A tensor of true labels.
      y_pred: A tensor of predicted labels.

    Returns:
      A PyTorch confusion matrix.
    """

    y_preds = []
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=16, shuffle=False, num_workers=os.cpu_count() - 1
    )

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(
                dim=1
            )  # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")

    confmat_tensor = confmat(
        preds=y_pred_tensor, target=torch.tensor(test_data.targets)
    )
        #plot confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.cpu().numpy(),
        class_names=class_names,
        figsize=(10, 7),
    )
    ax.set_xlabel("Predicted", fontsize=17)
    ax.set_ylabel("Actual", fontsize=17)
    fig.show()
    fig.savefig("confusion_matrix.png", bbox_inches="tight")
    return confmat_tensor.numpy()


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
