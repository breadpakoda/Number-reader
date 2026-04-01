"""
MNIST Deep Learning Classifier
================================
- Checks if a trained model exists; loads it if found, trains if not
- Data augmentation for improved generalization
- Accepts any image (color or greyscale), preprocesses it, and predicts the digit
"""

import os
import sys
import numpy as np

# ── Suppress TF verbose logs ──────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from PIL import Image, ImageOps, ImageFilter
except ImportError:
    print("[ERROR] Pillow is not installed. Run:  pip install Pillow")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH   = "mnist_model.keras"   # saved model path
IMAGE_SIZE   = (28, 28)              # MNIST canonical size
BATCH_SIZE   = 128
EPOCHS       = 30                    # max epochs (early-stopping kicks in earlier)
RANDOM_SEED  = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def build_model() -> keras.Model:
    """CNN with BatchNorm + Dropout for robust digit recognition."""
    inputs = keras.Input(shape=(28, 28, 1), name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation="softmax", name="digit")(x)

    model = keras.Model(inputs, outputs, name="mnist_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Download MNIST, normalise, and return train/val/test splits."""
    print("[INFO] Loading MNIST dataset ...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalise to [0, 1] and add channel dim
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    # Carve out 10 % validation set
    val_split = int(len(x_train) * 0.1)
    x_val, y_val       = x_train[:val_split], y_train[:val_split]
    x_train, y_train   = x_train[val_split:], y_train[val_split:]

    print(f"[INFO] Train: {len(x_train):,}  Val: {len(x_val):,}  Test: {len(x_test):,}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def make_augmentation_generator() -> ImageDataGenerator:
    """
    Real-world digits are rotated, shifted, and slightly zoomed.
    These transforms teach the network to handle such variance.
    """
    return ImageDataGenerator(
        rotation_range=12,          # random rotation +-12 degrees
        width_shift_range=0.12,     # horizontal shift up to 12 %
        height_shift_range=0.12,    # vertical shift up to 12 %
        zoom_range=0.15,            # zoom in/out +-15 %
        shear_range=0.1,            # slight shear
        fill_mode="nearest",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model: keras.Model, x_train, y_train, x_val, y_val):
    """Train with augmentation, early-stopping, and LR scheduling."""
    datagen = make_augmentation_generator()
    datagen.fit(x_train)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[INFO] Starting training (max {EPOCHS} epochs, early-stopping enabled) ...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(path: str) -> np.ndarray:
    """
    Load any image, convert to greyscale, resize to 28x28,
    invert if the background is light (MNIST uses white-on-black),
    and normalise to [0, 1].
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    # ── 1. Convert to greyscale ───────────────────────────────────────────────
    if img.mode != "L":
        print(f"[INFO] Converting '{img.mode}' image to greyscale ...")
        img = img.convert("L")
    else:
        print("[INFO] Image is already in greyscale format.")

    # ── 2. Resize with high-quality resampling ────────────────────────────────
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    # ── 3. Light denoising ────────────────────────────────────────────────────
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # ── 4. Enhance contrast so the digit stands out ───────────────────────────
    arr = np.array(img, dtype="float32")

    # Adaptive normalisation: stretch pixel range to [0, 255]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0

    # ── 5. Invert if background is light (MNIST: digit=white, bg=black) ───────
    if arr.mean() > 127:
        print("[INFO] Light background detected — inverting image ...")
        arr = 255.0 - arr

    # ── 6. Normalise to [0, 1] and add batch + channel dims ───────────────────
    arr = arr / 255.0
    arr = arr[np.newaxis, ..., np.newaxis]   # shape: (1, 28, 28, 1)
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_digit(model: keras.Model, image_path: str) -> None:
    """Preprocess the image and print the predicted digit with confidence."""
    print(f"\n[INFO] Preprocessing '{image_path}' ...")
    arr = preprocess_image(image_path)

    probs  = model.predict(arr, verbose=0)[0]     # shape: (10,)
    digit  = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100

    print("\n" + "=" * 45)
    print(f"  Predicted digit  :  {digit}")
    print(f"  Confidence       :  {confidence:.2f} %")
    print("=" * 45)

    # Show top-3 alternatives
    top3 = np.argsort(probs)[::-1][:3]
    print("\n  Top-3 predictions:")
    for rank, idx in enumerate(top3, 1):
        bar = "#" * int(probs[idx] * 30)
        print(f"    {rank}. Digit {idx}  {probs[idx]*100:6.2f} %  {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  MNIST Handwritten Digit Recogniser")
    print("=" * 55 + "\n")

    # ── Load or train model ───────────────────────────────────────────────────
    if os.path.isfile(MODEL_PATH):
        print(f"[INFO] Saved model found at '{MODEL_PATH}'. Loading ...")
        model = keras.models.load_model(MODEL_PATH)
        model.summary()

        # Quick sanity-check on the test set
        (_, _), (_, _), (x_test, y_test) = load_and_prepare_data()
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"[INFO] Test accuracy (loaded model): {acc*100:.2f} %\n")
    else:
        print(f"[INFO] No saved model found at '{MODEL_PATH}'. Training from scratch ...\n")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_prepare_data()

        model = build_model()
        model.summary()

        model, _ = train_model(model, x_train, y_train, x_val, y_val)

        # Final evaluation
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\n[INFO] Test accuracy (freshly trained): {acc*100:.2f} %")
        print(f"[INFO] Model saved to '{MODEL_PATH}'\n")

    # ── Interactive prediction loop ───────────────────────────────────────────
    print("-" * 55)
    print("  Enter the path to an image containing a handwritten digit.")
    print("  Supported formats: PNG, JPG, BMP, TIFF, WebP, etc.")
    print("  Type  'exit'  or  'quit'  to stop.")
    print("-" * 55)

    while True:
        try:
            image_path = input("\n  Image path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if image_path.lower() in {"exit", "quit", "q"}:
            print("[INFO] Goodbye!")
            break

        if not image_path:
            print("[WARN] No path entered. Please try again.")
            continue

        try:
            predict_digit(model, image_path)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
        except Exception as e:
            print(f"[ERROR] Could not process image: {e}")


if __name__ == "__main__":
    main()