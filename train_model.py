import os
import shutil
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_DIR = "model"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATH = os.path.join(MODEL_DIR, "fruitfour_mobilenetv1_fruit_condition.keras")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_fruitfour_mobilenetv1_fruit_condition.keras")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")


def clean_dataset(folder_path):
    allowed_extensions = (".jpg", ".jpeg", ".png", ".webp")
    corrupt_dir = os.path.join("dataset", "corrupt")

    if not os.path.exists(corrupt_dir):
        os.makedirs(corrupt_dir)

    total_checked = 0
    total_corrupt = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_checked += 1

            if not file.lower().endswith(allowed_extensions):
                total_corrupt += 1
                shutil.move(file_path, os.path.join(corrupt_dir, file))
                continue

            try:
                img = Image.open(file_path)
                img.verify()
            except Exception:
                total_corrupt += 1
                shutil.move(file_path, os.path.join(corrupt_dir, file))

    print("\n=== DATA CLEANING ===")
    print(f"Folder dicek       : {folder_path}")
    print(f"Total file dicek   : {total_checked}")
    print(f"File rusak/dipindah: {total_corrupt}")


def count_images(folder_path):
    counts = {}

    for class_name in CLASS_NAMES:
        class_path = os.path.join(folder_path, class_name)

        if os.path.exists(class_path):
            total = len([
                file for file in os.listdir(class_path)
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ])
            counts[class_name] = total
        else:
            counts[class_name] = 0

    return counts


clean_dataset(TRAIN_DIR)
clean_dataset(TEST_DIR)

train_counts = count_images(TRAIN_DIR)
test_counts = count_images(TEST_DIR)

print("\n=== ANALISIS DATA AWAL / EDA ===")
print("Jumlah data training:")
for class_name, total in train_counts.items():
    print(f"{class_name}: {total} gambar")

print("\nJumlah data testing:")
for class_name, total in test_counts.items():
    print(f"{class_name}: {total} gambar")


plt.figure(figsize=(10, 5))
plt.bar(train_counts.keys(), train_counts.values())
plt.title("Jumlah Data Training per Kelas")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Gambar")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_train_distribution.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(test_counts.keys(), test_counts.values())
plt.title("Jumlah Data Testing per Kelas")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Gambar")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_test_distribution.png"))
plt.close()


plt.figure(figsize=(12, 7))

for index, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(TRAIN_DIR, class_name)

    if not os.path.exists(class_path):
        continue

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    if len(image_files) > 0:
        sample_image = random.choice(image_files)
        img_path = os.path.join(class_path, sample_image)
        img = Image.open(img_path).convert("RGB").resize((224, 224))

        plt.subplot(2, 3, index + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eda_sample_images.png"))
plt.close()


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES,
    shuffle=False
)

with open(CLASS_INDEX_PATH, "w") as file:
    json.dump(train_generator.class_indices, file, indent=4)

print("\n=== PREPROCESSING ===")
print("Ukuran gambar       :", IMG_SIZE, "x", IMG_SIZE)
print("Batch size          :", BATCH_SIZE)
print("Kelas               :", train_generator.class_indices)
print("Jumlah kelas        :", len(CLASS_NAMES))
print("Augmentasi training : rotation, zoom, shift, shear, horizontal flip")


print("\n=== PEMODELAN CNN ===")
print("Model yang digunakan: MobileNetV1 Transfer Learning")
print("Output model        : 6 kelas buah dan kondisi")

base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(len(CLASS_NAMES), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop, checkpoint]
)

model.save(MODEL_PATH)

print("\nModel final berhasil disimpan di:")
print(MODEL_PATH)

print("\nModel terbaik berhasil disimpan di:")
print(BEST_MODEL_PATH)


print("\n=== PENGUJIAN DAN EVALUASI MODEL ===")

loss, accuracy = model.evaluate(test_generator)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)
print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

with open(os.path.join(MODEL_DIR, "classification_report_mobilenetv1.txt"), "w") as file:
    file.write("Classification Report MobileNetV1\n")
    file.write("=================================\n\n")
    file.write(report)
    file.write("\n\nConfusion Matrix\n")
    file.write("================\n")
    file.write(str(cm))
    file.write(f"\n\nTest Accuracy: {accuracy * 100:.2f}%\n")


print("\n=== VISUALISASI HASIL ===")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Testing/Validation Accuracy")
plt.title("Training dan Testing Accuracy MobileNetV1")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(MODEL_DIR, "accuracy_plot_mobilenetv1.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Testing/Validation Loss")
plt.title("Training dan Testing Loss MobileNetV1")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(MODEL_DIR, "loss_plot_mobilenetv1.png"))
plt.close()

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix MobileNetV1")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=30)
plt.yticks(np.arange(len(class_labels)), class_labels)

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_mobilenetv1.png"))
plt.close()

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

max_probs = np.max(predictions, axis=1)

print("\n=== ANALISIS CONFIDENCE ===")
print("Confidence minimum :", np.min(max_probs))
print("Confidence rata-rata :", np.mean(max_probs))
print("Confidence maksimum :", np.max(max_probs))

print("\nTraining selesai.")
print("File hasil tersimpan di folder model:")
print("- fruitfour_mobilenetv1_fruit_condition.keras")
print("- best_fruitfour_mobilenetv1_fruit_condition.keras")
print("- class_indices.json")
print("- eda_train_distribution.png")
print("- eda_test_distribution.png")
print("- eda_sample_images.png")
print("- accuracy_plot_mobilenetv1.png")
print("- loss_plot_mobilenetv1.png")
print("- confusion_matrix_mobilenetv1.png")
print("- classification_report_mobilenetv1.txt")
