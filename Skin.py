import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


# FUNCTION TO LOAD AND PREPROCESS IMAGE
def load_and_preprocess_images(directory, label, image_size=(128, 128), clahe_clip_limit=2.0):
    images = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, image_size)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit)
            img = clahe.apply(img)

            images.append(img)
            labels.append(label)

    return images, labels


# DATAPATH
base_path = 'C:\\Users\\musha\\Downloads\\archive (1)\\Oily-Dry-Skin-Types\\'
train_dry = os.path.join(base_path, 'train/dry')
train_normal = os.path.join(base_path, 'train/normal')
train_oily = os.path.join(base_path, 'train/oily')

test_dry = os.path.join(base_path, 'test/dry')
test_normal = os.path.join(base_path, 'test/normal')
test_oily = os.path.join(base_path, 'test/oily')

valid_dry = os.path.join(base_path, 'valid/dry')
valid_normal = os.path.join(base_path, 'valid/normal')
valid_oily = os.path.join(base_path, 'valid/oily')

# LOAD AND PREPROCESS DATA
print("Loading and Preprocessing Training Data:")
train_images = []
train_labels = []

dry_images, dry_labels = load_and_preprocess_images(train_dry, label='dry')
train_images.extend(dry_images)
train_labels.extend(dry_labels)

normal_images, normal_labels = load_and_preprocess_images(train_normal, label='normal')
train_images.extend(normal_images)
train_labels.extend(normal_labels)

oily_images, oily_labels = load_and_preprocess_images(train_oily, label='oily')
train_images.extend(oily_images)
train_labels.extend(oily_labels)

# Convert to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# LABEL ENCODING
label_encoder = LabelEncoder()
train_labels_numeric = label_encoder.fit_transform(train_labels)
train_labels_one_hot = to_categorical(train_labels_numeric)

# NORMALIZE IMAGE PIXEL VALUES
train_images_normalized = train_images / 255.0

# SPLIT DATA INTO TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    train_images_normalized, train_labels_one_hot, test_size=0.2, random_state=42
)

# CNN MODEL
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # 3 CLASSES (dry, normal, oily)

# COMPILE THE MODEL
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# SHOW MODEL SUMMARY
model.summary()


# TRAIN THE MODEL WITH DATA AUGMENTATION
batch_size = 32
epochs = 30

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow(X_train.reshape((-1, 128, 128, 1)), y_train, batch_size=batch_size)

history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_test.reshape((-1, 128, 128, 1)), y_test))


#PLOT TRAIN HISTORY
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# FUNCTION TO PREDICT SKIN TYPE FROM UPLOADED IMAGE
def predict_skin_type():
    global filename
    global prediction_label

    if filename:
        # Load and preprocess the uploaded image
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = img.reshape((1, 128, 128, 1))

        # Make prediction
        predicted_probabilities = model.predict(img)
        predicted_class = np.argmax(predicted_probabilities)
        predicted_class_label = label_encoder.classes_[predicted_class]

        # Update the prediction label
        prediction_label.config(text=f"Predicted Skin Type: {predicted_class_label}")
    else:
        prediction_label.config(text="Please upload an image first.")


# FUNCTION TO OPEN FILE DIALOG FOR IMAGE UPLOAD
def open_image():
    global filename
    filename = filedialog.askopenfilename(
        initialdir="/",
        title="Select a File",
        filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("all files", "*.*"))
    )

    if filename:
        # Display the uploaded image in the GUI
        img = Image.open(filename)
        img = img.resize((256, 256))  # Resize the image for display
        img = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img)
        uploaded_image_label.image = img  # Keep a reference to avoid garbage collection


# CREATE GUI
root = tk.Tk()
root.title("Skin Type Prediction")

# Upload Image Button
upload_button = tk.Button(root, text="Upload Skin Image", command=open_image)
upload_button.pack(pady=10)

# Uploaded Image Label
uploaded_image_label = tk.Label(root)
uploaded_image_label.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict Skin Type", command=predict_skin_type)
predict_button.pack(pady=10)

# Prediction Label
prediction_label = tk.Label(root, text="")
prediction_label.pack()

# Run the GUI
root.mainloop()