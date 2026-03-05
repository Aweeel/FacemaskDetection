import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

# Paths
dataset_dir = "dataset"  # should contain class folders
model_save_path = "models/face_obstruction_model2.h5"

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def augment_image(image):
    """
    Color-invariant augmentation that simulates diverse mask colors (black, pink,
    white, blue, etc.) by randomly shifting the brightness of the lower face region
    independently. CLAHE preserves structural edges regardless of mask color.
    Forces the model to learn structure (mask edge, covered lips/nose) not color.
    Input image is float32 in [0, 255] range (before rescale).
    """
    img = image.astype(np.uint8)
    h, w = img.shape[:2]

    # --- Random global brightness/contrast ---
    alpha = np.random.uniform(0.7, 1.3)
    beta  = int(np.random.randint(-30, 30))
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # --- Independently shift brightness of lower face (nose-to-chin) ---
    # This simulates black masks (darken), white/pink masks (brighten),
    # or any color mask relative to the upper face brightness.
    # Applied to BOTH classes so the model can't cheat using overall brightness.
    lower_y = int(h * 0.45)
    lower_region = img[lower_y:, :].astype(np.float32)
    lower_shift = np.random.uniform(-80, 80)  # wide range: very dark to very bright
    lower_region = np.clip(lower_region + lower_shift, 0, 255).astype(np.uint8)
    img[lower_y:, :] = lower_region

    # --- CLAHE on luminance to sharpen mask edges and facial structure ---
    # Works on the already-brightness-shifted image so edge sharpening applies
    # consistently regardless of whether lower face was made dark or bright
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    # --- Grayscale tiled to 3 channels: eliminates color as a signal ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.stack([gray, gray, gray], axis=-1)

    return img.astype(np.float32)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,

    brightness_range=(0.5, 1.5),
    channel_shift_range=60.0,

    horizontal_flip=True,
    fill_mode="nearest",

    preprocessing_function=augment_image,
    validation_split=0.2
)

# Train generator
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Build model using MobileNetV2 transfer learning
if os.path.exists(model_save_path):
    print(f"Loading existing model from {model_save_path} to continue training...")
    model = tf.keras.models.load_model(model_save_path)
    # Unfreeze the last 50 layers of the base model for continued fine-tuning
    base_model = model.layers[1]  # MobileNetV2 is the second layer (after Input)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
else:
    print("No existing model found, building from scratch...")
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze all but the last 50 layers — more layers unfrozen means it can learn
    # fine-grained texture (mask fabric weave, edge stitching) vs bare skin pores
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation="softmax")(x)  # 2 classes: with_mask, without_mask

    model = Model(inputs=base_model.input, outputs=output)

# Compile model — lower LR for fine-tuning upper layers
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Save model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
