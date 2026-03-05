import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset folder
dataset_dir = "dataset"  # folder containing class subfolders

# Setup ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create a generator to check classes
generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224), 
    batch_size=16,
    class_mode="categorical",
    subset="training",    
    shuffle=False
)

# Print class indices (folder -> integer label)
print("\nClass indices mapping:")
for class_name, index in generator.class_indices.items():
    class_folder = os.path.join(dataset_dir, class_name)
    num_images = len([f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"{class_name} -> {index} ({num_images} images)")

# Optional: show a few one-hot labels for the first batch
x_batch, y_batch = next(generator)
print("\nShape of first batch:", x_batch.shape, y_batch.shape)
print("One-hot labels for first batch:\n", y_batch)
