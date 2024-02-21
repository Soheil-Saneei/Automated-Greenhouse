import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense

from sklearn.preprocessing import LabelBinarizer
labelencoder = LabelBinarizer()
label = labelencoder.fit_transform([0,1,2,3])

image_shape = (256,256)
input_shape = (256,256,3)
folder = 'lettuce-8270'

# import the training data
x,y = [],[]
index = 0
for cat in os.listdir(folder + '/train/'):
  for image_path in os.listdir(folder + '/train/' + cat):
    img = mpimg.imread(folder + '/train/' + cat + '/' + image_path)
    # the images are all approximately of the same dimension (1024, 1024, 3),
    #  but there is some variation, so we need to resize them.
    img = cv2.resize(img, image_shape)
    if img.shape[2] == 4:
      img = img[:, :, :3]
    x.append(img)
    y.append(label[index])
  index += 1
print(len(x))
x_train = np.array(x, dtype=np.float16)
y_train = np.array(y, dtype=np.float16)

# import the validation data
x,y = [],[]
index = 0
for cat in os.listdir(folder + '/val/'):
  for image_path in os.listdir(folder + '/val/' + cat):
    img = mpimg.imread(folder + '/val/' + cat + '/' + image_path)
    img = cv2.resize(img, image_shape)
    x.append(img)
    y.append(label[index])
  index += 1
x_val = np.array(x, dtype=np.float32)
y_val = np.array(y, dtype=np.float32)

# define the neural network architecture
model = Sequential()

model.add(Conv2D(128, (3,3), input_shape=input_shape, activation="relu"))

model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))

model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))

model.summary()

# compile the model
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

# train the model
model.fit(
  x=x_train,
  y=y_train,
  validation_data=(x_val,y_val),
  batch_size=25,
  epochs=100
)

model.save('models/model_cnn2.keras')
