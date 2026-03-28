import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

CLASS_NAMES = [
    "stage0_unripe",
    "stage1_ripe",
    "stage2_overripe",
    "stage3_rotten"
]

# -------------------------------------------------
# CLASS COUNTS + WEIGHTS
# -------------------------------------------------
def get_class_counts(train_dir):
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(train_dir, cls)
        counts[cls] = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    return counts


def compute_class_weights(counts):
    total = sum(counts.values())
    n_classes = len(counts)
    return {
        i: total / (n_classes * counts[cls])
        for i, cls in enumerate(CLASS_NAMES)
    }

# -------------------------------------------------
# DATA PIPELINE
# -------------------------------------------------
def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(
        lambda x, y: (preprocess(data_aug(x, training=True)), y),
        num_parallel_calls=AUTOTUNE
    ).cache().shuffle(1000).prefetch(AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=AUTOTUNE
    ).cache().prefetch(AUTOTUNE)

    return train_ds, test_ds

# -------------------------------------------------
# MODEL
# -------------------------------------------------
def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    counts = get_class_counts(TRAIN_DIR)
    class_weights = compute_class_weights(counts)

    print("\n[INFO] Class distribution:")
    for k, v in counts.items():
        print(f"{k:18s}: {v}")

    train_ds, test_ds = build_datasets()
    model = build_model()
    model.summary()

    print("\n[INFO] Training started...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        class_weight=class_weights
    )

    # -------------------------------------------------
    # ACCURACY GRAPH (FOR PAPER)
    # -------------------------------------------------
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    epochs_range = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_acc, label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    acc_plot_path = os.path.join(DATASET_ROOT, "training_vs_validation_accuracy.png")
    plt.savefig(acc_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Accuracy graph saved at: {acc_plot_path}")

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------
    loss, acc = model.evaluate(test_ds)
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

    # -------------------------------------------------
    # SAVE MODELS
    # -------------------------------------------------
    final_model = os.path.join(DATASET_ROOT, "banana_stage_mobilenet_final.keras")
    model.save(final_model)
    print(f"[INFO] Model saved: {final_model}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(DATASET_ROOT, "banana_stage_mobilenet_final.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] TFLite saved: {tflite_path}")

# -------------------------------------------------
if __name__ == "__main__":
    main()
