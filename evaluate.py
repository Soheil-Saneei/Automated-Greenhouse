import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from sklearn.preprocessing import LabelBinarizer
labelencoder = LabelBinarizer()
label = labelencoder.fit_transform([0,1,2,3])

image_shape = (256,256)
folder = 'lettuce-8270'

def import_testing_data():
  x,y = [],[]
  index = 0
  for cat in os.listdir(folder + '/test/'):
    for image_path in os.listdir(folder + '/test/' + cat):
      img = mpimg.imread(folder + '/test/' + cat + '/' + image_path)
      img = cv2.resize(img, image_shape)
      if img.shape[2] == 4:
        img = img[:,:,:3]
      x.append(img)
      y.append(label[index])
    index += 1
  return (np.array(x, dtype=np.float32), np.array(y, dtype=np.float32))

def main():
  parser = argparse.ArgumentParser(
    prog='Evaluate',
    description='Evaluate a model on the testing data to determine the accuracy of prediction',
  )
  parser.add_argument('-m', '--model', default='1')
  args = parser.parse_args()

  # Import the testing data
  x_test, y_test = import_testing_data()
  # Load the model
  model = tf.keras.models.load_model('models/model_' + args.model + '.keras')

  # Evaluate the model on the training data
  loss, acc = model.evaluate(x_test, y_test)
  # Print the resulting accuracy
  print('Accuracy: {:.2f}%'.format(100*acc)) 

if __name__ == "__main__":
  main()
