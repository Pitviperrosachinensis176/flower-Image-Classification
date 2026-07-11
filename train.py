import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model import build_model


TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
MODEL_PATH = "flower_model.h5"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 5


def create_data_generators():

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,      
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator


def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("[INFO] Training plot saved as training_history.png.")


def main():
    print("=" * 60)
    print("Starting the flower classification model training process")
    print("=" * 60)

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"[ERROR] Dataset directories not found!")
        print(f"Please place the dataset in the following directories:")
        print(f"  - {TRAIN_DIR}")
        print(f"  - {VAL_DIR}")
        return

    print("\n[INFO] Loading data...")
    train_gen, val_gen = create_data_generators()

    print(f"[INFO] Number of classes: {train_gen.num_classes}")
    print(f"[INFO] Class names: {list(train_gen.class_indices.keys())}")
    print(f"[INFO] Number of training samples: {train_gen.samples}")
    print(f"[INFO] Number of validation samples: {val_gen.samples}")

    print("\n[INFO] Building CNN model...")
    model = build_model(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [

        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train the model
    print("\n[INFO] Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[INFO] Evaluating the final model...")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"[RESULT] Final accuracy on validation data: {val_acc * 100:.2f}%")
    print(f"[RESULT] Final loss: {val_loss:.4f}")

    class_indices = train_gen.class_indices
    with open('class_indices.txt', 'w', encoding='utf-8') as f:
        for cls, idx in class_indices.items():
            f.write(f"{idx}:{cls}\n")
    print("[INFO] Class mappings saved to class_indices.txt.")

    plot_history(history)

    print("\n" + "=" * 60)
    print(f"Training completed successfully. Model saved to {MODEL_PATH}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
