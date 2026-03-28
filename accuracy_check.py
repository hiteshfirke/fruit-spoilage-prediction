import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# 1. LOAD MODEL
# --------------------
model_path = "dataset/banana_stage_mobilenet_final.keras"
model = load_model(model_path)
print("[INFO] Model loaded:", model_path)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",   # NOT sparse_categorical_crossentropy
    metrics=["accuracy"]
)
# --------------------
# 2. LOAD TEST DATA
# --------------------
test_dir = "dataset/test"

datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    class_mode="categorical"
)

# --------------------
# 3. EVALUATE ACCURACY
# --------------------
loss, acc = model.evaluate(test_gen)
print("\n====================")
print(" MODEL ACCURACY REPORT")
print("====================")
print(f"Test Accuracy: {acc * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --------------------
# 4. CONFUSION MATRIX
# --------------------
pred_probs = model.predict(test_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes

labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --------------------
# 5. CLASSIFICATION REPORT
# --------------------
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=labels))
