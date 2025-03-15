# Image Classifier using TensorFlow

## Introduction
This project is a binary image classifier built using **TensorFlow** and **Keras** in a **WSL (Windows Subsystem for Linux) environment** with **VS Code**. The model classifies images into two categories: "Happy" and "Sad."

## Setup Instructions
### 1. Install WSL and VS Code
- Install **WSL** (Windows Subsystem for Linux) on Windows.
- Install **VS Code** and the **WSL extension**.
- Set up **Ubuntu** as your WSL distribution.

### 2. Install Python and Virtual Environment
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y
```

### 3. Install TensorFlow on WSL
1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv my_tf_env
   source my_tf_env/bin/activate
   ```
2. **Upgrade pip and Install TensorFlow**:
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install tensorflow numpy opencv-python matplotlib
   ```
3. **Verify TensorFlow Installation**:
   ```bash
   python3 -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Code Explanation

### 1. **Importing Required Libraries**
```python
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
```
- **TensorFlow**: Used for building and training the deep learning model.
- **OpenCV (`cv2`)**: Used for image preprocessing.
- **Matplotlib**: Used for visualization.
- **NumPy**: Used for numerical computations.

---

### 2. **GPU Configuration**
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
- Lists available GPUs and enables memory growth to prevent TensorFlow from consuming all GPU memory.

---

### 3. **Preprocessing the Data**
```python
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'Image not in ext list {image_path}')
                os.remove(image_path)
        except Exception as e:
            print(f'Issue with image {image_path}')
```
- Reads images from `data/` and removes any non-image files.
- Ensures only valid image formats (`jpeg, jpg, bmp, png`) are used.

---

### 4. **Creating a TensorFlow Dataset**
```python
data = tf.keras.utils.image_dataset_from_directory('data')
data = data.map(lambda x, y: (x / 255, y))
```
- Loads images into a **TensorFlow dataset**.
- Normalizes pixel values to [0,1] range for better model performance.

---

### 5. **Splitting Data into Training, Validation, and Test Sets**
```python
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
```
- **70%** Training, **20%** Validation, **10%** Testing.

---

### 6. **Building the CNN Model**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
- Uses **Convolutional Layers** to extract features from images.
- **MaxPooling** layers downsample images to reduce computation.
- **Dense layers** classify images into `Happy` or `Sad`.
- **Sigmoid Activation** is used for binary classification.

---

### 7. **Compiling and Training the Model**
```python
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
logsdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logsdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```
- **Adam Optimizer** is used for efficient learning.
- **Binary Crossentropy Loss** is used for binary classification.
- **TensorBoard** is used for monitoring training performance.

---

### 8. **Plotting Training Results**
```python
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
```
- Plots loss and accuracy trends to analyze model performance.

---

### 9. **Evaluating the Model**
```python
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(pre.result().numpy(), re.result().numpy(), acc.result().numpy())
```
- Computes **Precision, Recall, and Accuracy** for test data.

---

### 10. **Making a Prediction**
```python
img = cv2.imread('test_image.png')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize / 255, 0))
print(f'Predicted class is {"Sad" if yhat > 0.5 else "Happy"}')
```
- Loads an image, resizes it, and predicts its class using the trained model.

---

## **Conclusion**
This project implements an image classification model using TensorFlow in a WSL environment. It performs data preprocessing, builds a CNN, trains it, and evaluates its performance. The model successfully predicts images as either "Happy" or "Sad."

