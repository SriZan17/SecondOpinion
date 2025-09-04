import torch
from trainer_vit import create_vit_model
from going_modular.utils import calculate_confusion_matrix

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create ViT model and get transforms
    vit_model, vit_transforms, _ = create_vit_model(num_classes=4, seed=43)
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    # Load the saved ViT model weights
    vit_model.load_state_dict(torch.load("Models/tumor9542.pth", map_location=device))
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



if __name__ == "__main__":
    main()
