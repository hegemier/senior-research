import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

'''
This file trains the classification model used in the edge server.
'''

batch_size = 32
img_height = 240 #we used low resolution images from the pi camera for speed, but these can be adjusted
img_width = 240

data_dir = 'objectimages/'

#create training set

trainingSet = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=5,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#create validation set

validationSet = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=5,
  image_size=(img_height, img_width),
  batch_size=batch_size)

classNames = trainingSet.class_names
print("Classes are: " + str(classNames))

AUTOTUNE = tf.data.AUTOTUNE

trainingSet = trainingSet.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validationSet = validationSet.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

num_classes = 3

#data augmentation section

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print("training model...")

epochs = 3
history = model.fit(
  trainingSet,
  validation_data=validationSet,
  epochs=epochs
)

print("model trained.")

#save our model

import os
os.mkdir('saved_model_final', mode=0o777)
model.save('saved_model_final/my_model')

print("model saved")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochsRange = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochsRange, acc, label='Training Accuracy')
plt.plot(epochsRange, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochsRange, loss, label='Training Loss')
plt.plot(epochsRange, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('output_final.png') #some information on the model
