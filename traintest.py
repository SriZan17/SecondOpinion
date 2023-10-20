import os
import random
import shutil

# Set the path to your images directory
images_dirs = [
    "data/glioma_tumor",
    "data/meningioma_tumor",
    "data/normal",
    "data/pituitary_tumor",
]


for images_dir in images_dirs:
    # Set the percentage of images to use for testing
    test_percent = 0.25
    train_dir = ""
    test_dir = ""

    # Create the train and test directories
    train_dir = os.path.join(images_dir, "train")
    test_dir = os.path.join(images_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all the image filenames
    image_filenames = os.listdir(images_dir)

    # Shuffle the filenames
    random.shuffle(image_filenames)

    # Calculate the number of images to use for testing
    num_test_images = int(len(image_filenames) * test_percent)

    # Move the images to the test directory
    for i in range(num_test_images):
        filename = image_filenames[i]
        src_path = os.path.join(images_dir, filename)
        dst_path = os.path.join(test_dir)
        shutil.move(src_path, dst_path)

    # Move the remaining images to the train directory
    for filename in image_filenames[num_test_images:]:
        src_path = os.path.join(images_dir, filename)
        dst_path = os.path.join(train_dir)
        shutil.move(src_path, dst_path)
