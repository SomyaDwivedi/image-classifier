# Image Classification using TensorFlow and CNNs

## Overview
This project implements an image classifier using TensorFlow and Convolutional Neural Networks (CNNs). The classifier is trained to distinguish between two classes: "Happy" and "Sad" images.

## Project Structure
```
image-classifier/
│-- data/                # Dataset folder (contains class subfolders with images)
│-- models/              # Saved trained models
│-- logs/                # TensorBoard logs
│-- imageclassifier.py    # Main script for training and evaluation
│-- test_image.png        # Sample image for inference
│-- README.md             # Project documentation
```

## Installation
Ensure you have the following dependencies installed:
```bash
python3 -m pip install tensorflow numpy matplotlib seaborn opencv-python scikit-learn
```

## Dataset Preparation
The dataset should be structured as follows:
```
data/
│-- Happy/
│   ├── image1.jpg
│   ├── image2.jpg
│-- Sad/
│   ├── image1.jpg
│   ├── image2.jpg
```

## Code Explanation

### 1. Importing Required Libraries
The script imports necessary libraries such as TensorFlow, OpenCV, NumPy, and Matplotlib.
```python
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
```

### 2. Configuring GPU for TensorFlow
This ensures that TensorFlow efficiently utilizes GPU resources.
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
```

### 3. Loading and Filtering Dataset
The script iterates through the dataset directory, loads images, and removes unsupported formats.
```python
data_dir = 'data'
image_exts = ['jpeg','jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
```

### 4. Creating TensorFlow Dataset
A TensorFlow dataset is created from the directory.
```python
data = tf.keras.utils.image_dataset_from_directory('data')
```

### 5. Visualizing a Few Images
The script plots some sample images from the dataset.
```python
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
```

### 6. Normalizing Data
The images are normalized to scale pixel values between 0 and 1.
```python
data = data.map(lambda x,y: (x/255, y))
```

### 7. Splitting Data into Train, Validation, and Test Sets
```python
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
```

### 8. Building the CNN Model
The model consists of convolutional, max-pooling, and dense layers.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
```

### 9. Compiling and Training the Model
The model is compiled and trained for 20 epochs.
```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
logsdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logsdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```

### 10. Plotting Accuracy and Loss Graphs
```python
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
```

### 11. Evaluating the Model
Precision, recall, and accuracy are computed.
```python
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X) 
    yhat_labels = np.argmax(yhat, axis=1) 
    pre.update_state(y, yhat_labels)
    re.update_state(y, yhat_labels)
    acc.update_state(y, yhat_labels)
print(f'Precision: {pre.result().numpy()}')
print(f'Recall: {re.result().numpy()}')
print(f'Accuracy: {acc.result().numpy()}')
```

### 12. Making Predictions on a New Image
```python
img = cv2.imread('test_image.png')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print("Predicted class is Sad" if np.argmax(yhat) == 1 else "Predicted class is Happy")
```

### 13. Generating a Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = np.concatenate([y.numpy() for _, y in test_dataset])
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
```

## Author
Developed by Somya Dwivedi

