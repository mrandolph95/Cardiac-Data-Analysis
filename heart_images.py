# -*- coding: utf-8 -*-
"""Heart Images.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pMwdWi0VVHJdRD2PjB1j8YBm6aQZnA_N
"""

pip install tensorflow

pip install pydicom

pip install Pillow

import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import seaborn as sns
import pydicom
import pandas as pd
import random
from tensorflow.keras.utils import to_categorical

from PIL import Image

import zipfile

"""#### Unzip files"""

!unzip /content/SCD_IMAGES_05.zip

!unzip /content/SCD_IMAGES_01.zip

!unzip /content/SCD_IMAGES_03.zip

"""#### Pre-process images (adjust sizing, convert to JPEG)

Pre-process images and use supplemental csv file with patient data to categorize images by pathologies: heart failure with infarction, heart failure without infarction, hypertrophy, normal
"""

# Map labels in CSV file
def category_mapping(csv_file):
    mapping = {}
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
      patient_id = row["PatientID"]
      pathology = row["Pathology"]

      if 'Heart failure with infarct' in pathology:
        mapping[patient_id] = "HF-I"
      elif 'Heart failure without infarct' in pathology:
        mapping[patient_id] = "HF"
      elif 'Hypertrophy' in pathology:
        mapping[patient_id] = "HYP"
      elif 'Normal' in pathology:
        mapping[patient_id] = "N"
    return mapping

csv_file = "/content/scd_patientdata.csv"
patient_category_map = category_mapping(csv_file)

categories = ["HF-I", "HF", "HYP", "N"]
category_map = {category: idx for idx, category in enumerate(categories)}

# Process images
def process_dicom(dicom_root, image_size=(224, 224), output_folder="/content/output_jpegs"):
    images = []
    labels = []

    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(dicom_root):
        if not files:
            continue

        parent_folder_name = os.path.basename(os.path.dirname(root))

        if parent_folder_name not in patient_category_map:
            print(f"Skipping folder {parent_folder_name}: No category mapping found.")
            continue

        category = patient_category_map[parent_folder_name]
        label = category_map[category]

        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                try:
                    dicom = pydicom.dcmread(file_path)

                    if hasattr(dicom, "pixel_array"):
                        image = dicom.pixel_array

                        resized_image = cv2.resize(image, image_size)
                        img = np.clip(resized_image, 0, 255).astype(np.uint8)

                        save_subfolder = os.path.join(output_folder, category)
                        os.makedirs(save_subfolder, exist_ok=True)

                        filename = file.replace(".dcm", ".jpg")
                        save_path = os.path.join(save_subfolder, filename)
                        Image.fromarray(img).convert("L").save(save_path)

                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"Skipping {file_path}: No pixel data found")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if not images:
        print("No images were processed.")
        return [], []

    print(f"Processed {len(images)} images.")
    return np.array(images).reshape(-1, 224, 224, 1), np.array(labels)

images_1, labels_1 = process_dicom("/content/Heart Files_1")

print(f"Number of images: {len(images_1)}")
print(f"Number of labels: {len(labels_1)}")

from collections import Counter
label_counts = Counter(labels_1)
for category, count in label_counts.items():
    print(f"{categories[category]}: {count}")

images_2, labels_2 = process_dicom("/content/Heart Files_2")

print(f"Number of images: {len(images_2)}")
print(f"Number of labels: {len(labels_2)}")

from collections import Counter
label_counts = Counter(labels_2)
for category, count in label_counts.items():
    print(f"{categories[category]}: {count}")

images_3, labels_3 = process_dicom("/content/Heart Files_3")

print(f"Number of images: {len(images_3)}")
print(f"Number of labels: {len(labels_3)}")

from collections import Counter
label_counts = Counter(labels_3)
for category, count in label_counts.items():
    print(f"{categories[category]}: {count}")

# Merge all image and label samples
images = np.concatenate([images_1, images_2, images_3], axis=0)
labels = np.concatenate([labels_1, labels_2, labels_3], axis=0)

num_classes = len(category_map)
labels = to_categorical(labels, num_classes=num_classes)

labels_int = np.argmax(labels, axis=1)

# Reduce sample size to 2,500 per category to reduce overfitting
image_category = 10_000 // num_classes

image_samples = []
label_samples = []

# Normalize images
for category in range(num_classes):
    category_indices = np.where(labels_int == category)[0]
    sampled_indices = random.sample(list(category_indices), min(image_category, len(category_indices)))
    image_samples.append(images[sampled_indices])
    label_samples.extend(labels[sampled_indices])

image_samples = np.vstack(image_samples).astype("float32") / 255.0
label_samples = np.vstack(label_samples)

# Reshape images for model
image_samples = image_samples.reshape(-1, 224, 224, 1)

"""#### Display image from each category"""

category_images = {category: [] for category in categories}

for img, label in zip(image_samples, label_samples):
    label_index = np.argmax(label)
    category_images[categories[label_index]].append(img)

num_categories = len(categories)
fig, axes = plt.subplots(1, num_categories, figsize=(15, 5))

if num_categories == 1:
    axes = [axes]

for i, category in enumerate(categories):
    img = category_images[category][0]
    img = np.squeeze(img)

    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"Label: {category}")

plt.show()

"""### Build Neural Network"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, UpSampling2D, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""#### Build Model"""

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

"""#### Train Model"""

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_samples, label_samples, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

"""#### Save model to Google drive"""

from google.colab import drive
model.save('/content/drive/MyDrive/heart_model.h5')

"""### Predict"""

predictions_vector = model.predict(X_test)
predictions = np.argmax(predictions_vector, axis=1)

predicted_label_name = categories[predictions[9]]
true_label = np.argmax(y_test[9])
true_label_name = categories[true_label]

plt.imshow(X_test[0], cmap='gray')
plt.axis('off')
plt.show()
print("The predicted label is:", predicted_label_name)
print("The true label is:", true_label_name)