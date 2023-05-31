import os
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras_preprocessing.image import img_to_array
# Done collaboratively by Adam and Michael
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sklearn.preprocessing import LabelEncoder
import glob
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# use of tensorflow gpu taken from this video https://www.youtube.com/watch?v=EmZZsy7Ym-4&ab_channel=DerekBanas
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.datasets import cifar10
tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# image loading and resizing done by adam
# initial parameters
data = []
labels = []

# load images shuffled
image_files = [f for f in glob.glob(r'C:\project\data' + "/********/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)
label_encoder = LabelEncoder()

# converting images to arrays and labelling the categories
for img in image_files:
    #converts image into an array for the model
    image = cv2.imread(img)
    image = cv2.resize(image, (32,32))
    image = img_to_array(image)
    data.append(image)




    label = img.split(os.path.sep)[-2] # would look at the word anger since it's the 2nd to lastC:\Users\coolm\OneDrive\Documents\coding\ai project\faces\anger\image0000006.jpg
    labels.append([label]) # adds labels
labels = label_encoder.fit_transform(labels)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# model done by michael and regularization by tutorial https://www.youtube.com/watch?v=JEWzWv1fBFQ&ab_channel=JeffHeaton
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        128, 3, padding="same", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01),)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(8)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = my_model()
model.compile
(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)

model.fit(data, labels, batch_size=128, epochs=700, verbose=2)
model.evaluate(data, labels, batch_size=128, verbose=2)
model.save('Smile_Detection.model')
