import os
from PIL import Image

# Paths
dataset_dir = "dataset"  # original dataset
output_dir = "dataset_processed"  # new folder for processed images
target_size = (224, 224)  # size for training

# Loop over each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Make corresponding output folder
    out_class_path = os.path.join(output_dir, class_name)
    os.makedirs(out_class_path, exist_ok=True)

    # Process images
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for i, file_name in enumerate(images, start=1):
        old_path = os.path.join(class_path, file_name)
        ext = os.path.splitext(file_name)[1]

        # Open and convert to RGB
        img = Image.open(old_path).convert("RGB")

        # Resize using modern Pillow
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Save to new folder with sequential name
        new_name = f"{class_name}_{i:03d}.jpg"  # save as JPG
        new_path = os.path.join(out_class_path, new_name)
        img.save(new_path)

print("All images converted to RGB and resized!")
