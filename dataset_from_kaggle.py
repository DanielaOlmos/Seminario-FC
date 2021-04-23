import os
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


os.system(" kaggle datasets download -d moltean/fruits -p $PWD")
os.system(" unzip fruits.zip") 

import pathlib
data_dir_train = pathlib.Path("fruits-360/Training")
data_dir_test = pathlib.Path("fruits-360/Test")

image_count = len(list(data_dir_train.glob('*/*.jpg')))
print(image_count)

Banana = list(data_dir_train.glob('Banana/*'))
PIL.Image.open(str(Banana[0]))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
#  validation_split=0.2,
#  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_test,
#  validation_split=0.2,
#  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)





