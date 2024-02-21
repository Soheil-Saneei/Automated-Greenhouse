import os
import shutil
import numpy as np
from glob import glob

np.random.seed(42)

rng = np.random.default_rng()

# define the categories
categories = ['FN', '-N', '-P', '-K']

# make the folders
# can technically fail if randomly generated name is in use
folder_name = 'lettuce-' + str(np.random.randint(1e3,1e4))
os.mkdir(folder_name)
os.mkdir(folder_name + '/train/')
os.mkdir(folder_name + '/val/')
os.mkdir(folder_name + '/test/')
for cat in categories:
  os.mkdir(folder_name + '/train/' + cat)
  os.mkdir(folder_name + '/val/' + cat)
  os.mkdir(folder_name + '/test/' + cat)

# for each category, split of the indices and copy the files

for cat in categories:
  # compute the quantities of training, validation, and testing images
  filepaths = glob('FNNPK/' + cat + '/*')
  num = len(filepaths)
  num_train_val = int(0.8 * num)
  num_train = int(0.85 * num_train_val)

  # use random indices to sort files into the three groups
  train_val_indices = rng.choice(num, size=num_train_val, replace=False)
  train_indices = train_val_indices[:num_train]
  val_indices = train_val_indices[num_train:]
  test_indices = list(set(range(num)) - set(train_val_indices))

  # copy the files over
  for i in train_indices:
    filepath = filepaths[i]
    file_name = filepath.split('/')[-1] # just the file name, not the path
    shutil.copy(filepath, folder_name + '/train/' + cat + '/' + file_name)
  for i in val_indices:
    filepath = filepaths[i]
    file_name = filepath.split('/')[-1]
    shutil.copy(filepath, folder_name + '/val/' + cat + '/' + file_name)
  for i in test_indices:
    filepath = filepaths[i]
    file_name = filepath.split('/')[-1]
    shutil.copy(filepath, folder_name + '/test/' + cat + '/' + file_name)
