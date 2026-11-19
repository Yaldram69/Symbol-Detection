import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths to dataset and annotations
images_dir = './dataset/archive (16)/flickr_logos_27_dataset/flickr_logos_27_dataset_images'
train_annotations_file = './dataset/archive (16)/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt'

# Load and preprocess data
def load_data(images_dir, annotations_file):
    images = []
    labels = []
    with open(annotations_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            file_name, label = parts[0], parts[1]
            img_path = os.path.join(images_dir, file_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))  # Resize for MobileNetV2
                images.append(img)
                labels.append(label)

    images = np.array(images, dtype="float32") / 255.0
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = np.array(labels)
    return images, labels, le

# Load dataset and split into train and validation
X, y, label_encoder = load_data(images_dir, train_annotations_file)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
data_gen = ImageDataGenerator(rotation_range=15, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Load MobileNetV2 with pre-trained weights and add custom layers
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model layers

# Add custom layers on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callbacks for learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(data_gen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=30, callbacks=[reduce_lr, early_stop])

# Save the model and label encoder
model.save("mobilenet_logo_detection_model.h5")
np.save("label_encoder.npy", label_encoder.classes_)
print("Enhanced MobileNet model and label encoder saved.")
