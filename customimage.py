import torchvision
import torch
import pathlib
import matplotlib.pyplot as plt


def prepare_image(path_to_image):
    custom_image_path = pathlib.Path(path_to_image)

    # custom_image_path = pathlib.Path("example.jpg")

    # Load in custom image and convert the tensor values to float32
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
    if custom_image.shape[0] == 4:
        custom_image = custom_image[:3]

    # Divide the image pixel values by 255 to get them between [0, 1] if necessary
    custom_image = custom_image / 255.0

    # Perfrom necessary transformations
    custom_image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(64, 64)),
            # torchvision.transforms.ToTensor(),
        ]
    )
    custom_image_transformed = custom_image_transform(custom_image)

    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    return custom_image_transformed_with_batch_size


# custom_image_path = pathlib.Path("loss_curves.png")
#
## Load in custom image and convert the tensor values to float32
# custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
#
## Divide the image pixel values by 255 to get them between [0, 1] if necessary
# custom_image = custom_image / 255.0
#
## Perfrom necessary transformations
# custom_image_transform = torchvision.transforms.Compose(
#    [
#        torchvision.transforms.Resize(size=(64, 64)),
#    ]  # , torchvision.transforms.ToTensor()]
# )
# custom_image_transformed = custom_image_transform(custom_image)
#
## Add an extra dimension to image
# custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
#
## Print out image data
# print(f"Custom image tensor:\n{custom_image}\n")
# print(f"Custom image shape: {custom_image.shape}\n")
# print(f"Custom image dtype: {custom_image.dtype}")
# print(f"Original shape: {custom_image.shape}")
# print(f"New shape: {custom_image_transformed.shape}")
#
#
# plt.imshow(custom_image_transformed.permute(1, 2, 0))
# plt.show()
#
