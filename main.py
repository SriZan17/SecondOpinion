import torch
from dataprep import TinyVGG
from customimage import prepare_image


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_names = ["glioma_tumor", "meningioma_tumor", "normal", "pituitary_tumor"]
    model = TinyVGG(
        input_shape=3,  # number of color channels (3 for RGB)
        hidden_units=10,
        output_shape=4,
    ).to(DEVICE)
    model.load_state_dict(torch.load("Models/model_2.pth"))
    model.eval()
    image = prepare_image("loss_curves.png")
    prediction = model(image.to(DEVICE))
    pred_prob = torch.softmax(prediction, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    print(f"Predicted label: {label_names[pred_label.item()]}")
    print(f"Predicted probability: {pred_prob.max().item():.3f}")


if __name__ == "__main__":
    main()
