import torch
from trainer_vit import create_vit_model
from going_modular.utils import calculate_confusion_matrix, collect_predictions, plot_multiclass_roc
from torch.utils.data import DataLoader
from torchvision import datasets

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create ViT model and get transforms
    vit_model, vit_transforms, _ = create_vit_model(num_classes=4, seed=43)
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    # Load the saved ViT model weights
    vit_model.load_state_dict(torch.load("Models/best_model.pth", map_location=device))
    print("Model loaded successfully!")
    
    # Set class names for brain tumor classification
    class_names = [
        "Glioma Tumor",
        "Meningioma Tumor", 
        "Normal Brain",
        "Pituitary Tumor"
    ]
    
    # Set test directory path
    test_dir = "Data/Tumors/test"

    # Existing confusion matrix
    confmat_tensor = calculate_confusion_matrix(
        class_names=class_names,
        model=vit_model,
        test_dir=test_dir,
        device=device,
        transform=vit_transforms,
    )
    print("Confusion matrix saved as 'confusion_matrix_vit.png'")
    print("Confusion Matrix:")
    print(confmat_tensor)

    # Build test DataLoader for ROC
    test_dataset = datasets.ImageFolder(root=test_dir, transform=vit_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Collect predictions
    y_true, y_score = collect_predictions(vit_model, test_loader, device)

    # Plot ROC
    plot_multiclass_roc(
        y_true=y_true,
        y_score=y_score,
        class_names=class_names,
        save_path="roc_curve_vit.png"
    )

if __name__ == "__main__":
    main()
