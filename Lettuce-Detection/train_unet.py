import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import keras

from keras import Input
from keras.models import Functional
from keras import layers

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

# helper functions to define useful blocks of layers

def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x


def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)

   return f, p


def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

# define the neural network architecture

inputs = Input(shape=input_shape)

f1, p1 = downsample_block(inputs, 32)

f2, p2 = downsample_block(p1, 64)

f3, p3 = downsample_block(p2, 128)

f4, p4 = downsample_block(p3, 256)

bottleneck = double_conv_block(p4, 512)

u6 = upsample_block(bottleneck, f4, 256)

u7 = upsample_block(u6, f3, 128)

u8 = upsample_block(u7, f2, 64)

u9 = upsample_block(u8, f1, 32)


flat = layers.Flatten()(u9)
dense1 = layers.Dense(256, activation="relu")(flat)
outputs = layers.Dense(4, activation="softmax")(dense1)

model = keras.Model(inputs=inputs, outputs=outputs)

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
  epochs=30
)

model.save('models/model_unet.keras')
