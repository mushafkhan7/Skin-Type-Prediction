Sure! Here's the entire content with headings in Markdown format:

```markdown
# Skin Type Prediction

Skin Type Prediction is a Python application that predicts the skin type (dry, normal, or oily) based on uploaded images using convolutional neural networks (CNNs). This project leverages the TensorFlow and scikit-learn libraries for model building and evaluation.

## Code Explanation

### Loading and Preprocessing Images

```python
import os
import cv2
import numpy as np

# Function to load and preprocess images
def load_and_preprocess_images(directory, label, image_size=(128, 128), clahe_clip_limit=2.0):
    # Implementation details...
```

This function loads and preprocesses images from a specified directory.

### Data Paths 

```python
base_path = 'C:\\Users\\username\\Downloads\\archive (1)\\Oily-Dry-Skin-Types\\'
train_dry = os.path.join(base_path, 'train/dry')
train_normal = os.path.join(base_path, 'train/normal')
train_oily = os.path.join(base_path, 'train/oily')
```

These are the paths to the directories containing the training data for dry, normal, and oily skin types.

### Loading and Preprocessing Data

```python
train_images = []
train_labels = []

dry_images, dry_labels = load_and_preprocess_images(train_dry, label='dry')
train_images.extend(dry_images)
train_labels.extend(dry_labels)

# Similar loading and preprocessing for normal and oily skin types
```

This code segment loads and preprocesses training images for each skin type using the previously defined function.

### Model Definition and Compilation

```python
from tensorflow.keras import layers, models

model = models.Sequential()

# CNN model definition...
```

This code defines a convolutional neural network (CNN) model for skin type prediction and compiles it with Adam optimizer and categorical cross-entropy loss.

### Training the Model with Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Data augmentation parameters...
```

This code segment defines an ImageDataGenerator for data augmentation and trains the model using augmented data.

### Plotting Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

This code plots the training and validation accuracies over epochs to visualize the training history.

### GUI for Predictions

```python
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# GUI initialization and elements definition...
```

This code initializes a graphical user interface (GUI) using tkinter for uploading images and displaying predictions interactively.

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- tkinter
- OpenCV
- Pillow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mushafkhan7/skin-type-prediction.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the project directory:

   ```bash
   cd skin-type-prediction
   ```

2. Run the application:

   ```bash
   python skin_type_prediction.py
   ```

3. Upload a skin image using the GUI.
4. Click on the "Predict Skin Type" button to obtain the predicted skin type.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss the proposed changes.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [OpenCV](https://opencv.org/) for image processing capabilities.
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework.
- [scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [matplotlib](https://matplotlib.org/) for data visualization.
- [tkinter](https://docs.python.org/3/library/tkinter.html) for GUI development.
- [Pillow](https://python-pillow.org/) for image processing in Python.

## Authors

- **Mushaf Khan** - *Initial work* - [GitHub Profile](https://github.com/mushafkhan7)
```
