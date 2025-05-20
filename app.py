import random
from pathlib import Path
from timeit import default_timer as timer

import gradio as gr
import torch
from torchcam.methods import GradCAM  # gradient-based CAM extractor
from torchcam.utils import overlay_mask  # overlay utility for heatmaps
from torchvision.transforms.functional import to_pil_image

from trainer_vit import create_vit_model


def main():
    # Get a list of all test image filepaths
    test_dir = "Data/Tumors/test"
    test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

    # Create a list of example inputs to our Gradio demo
    example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

    title = "Tumor Classifier with Grad-CAM"
    description = (
        "A Vision-Transformer based tumor classifier that highlights " "regions driving its predictions using Grad-CAM."
    )
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=3, label="Predictions"),
            gr.Number(label="Prediction time (s)"),
            gr.Image(label="CAM Heatmap"),
        ],
        examples=example_list,
        title=title,
        description=description,
    )
    demo.launch(debug=False)


def predict(img):
    """Transforms and performs prediction + Grad-CAM, returning (preds, time, overlay)."""
    start_time = timer()

    device = "cpu"
    class_names = [
        "Glioma Tumor",
        "Meningioma Tumor",
        "Normal Brain",
        "Pituitary Tumor",
    ]

    # Load model & transforms
    model, model_transforms = create_vit_model(num_classes=len(class_names))
    model.load_state_dict(torch.load("Models/vit.pth", map_location=device))
    model.to(device)

    # Store the original requires_grad state of parameters (important for some models)
    original_param_states = {}
    for name, param in model.named_parameters():
        original_param_states[name] = param.requires_grad
        # Ensure that parameters contributing to conv_proj have gradients enabled
        # This is a general approach; for ViT, conv_proj is usually the patch embedding conv.
        if "conv_proj" in name:  # You might need to adjust this check based on your ViT's exact structure
            param.requires_grad = True

    # Prepare CAM extractor (adjust target_layer as needed)
    cam_extractor = GradCAM(model, target_layer=model.conv_proj)

    # Preprocess & inference
    img_tensor = model_transforms(img).unsqueeze(0).to(device)

    # Temporarily set the model to training mode to ensure gradient tracking
    # for the necessary layers, then revert to evaluation mode.
    model.train()  # Enable gradients for the forward pass for CAM

    with torch.enable_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    model.eval()  # Set model back to eval mode for standard inference if desired

    # Restore original requires_grad states (good practice if you're not done with the model)
    for name, param in model.named_parameters():
        if name in original_param_states:
            param.requires_grad = original_param_states[name]

    # Build prediction dict
    pred_dict = {class_names[i]: float(probs[0, i]) for i in range(len(class_names))}
    if max(pred_dict.values()) < 0.52:
        pred_dict = {"Unknown": 1.0}

    # Extract and process CAM
    class_idx = outputs.argmax(dim=1).item()
    # The cam_extractor uses the outputs to backpropagate to the target_layer
    activation_map = cam_extractor(class_idx, outputs)[0].squeeze(0)
    mask_img = to_pil_image(activation_map, mode="F")
    overlay = overlay_mask(img, mask_img, alpha=0.5)

    elapsed = round(timer() - start_time, 5)
    return pred_dict, elapsed, overlay


if __name__ == "__main__":
    main()
