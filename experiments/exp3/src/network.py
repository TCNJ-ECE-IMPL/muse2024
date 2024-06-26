import math
import numpy as np
import os
import nptdms
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import network_functions as nf
import keras

############
# CONSTANTS:
############
INPUT_SIZE = 1024
USE_FFT = False

#######
# MAIN:
#######

# Get dataset from tdms files
dataset = nf.get_base_dataset("../tdms_data/")
# Shuffle dataset
random.shuffle(dataset)

# Partition dataset
train_frac = 0.7
valid_frac = 0.2

train_end_index = math.floor(len(dataset) * train_frac)
valid_end_index = train_end_index + math.floor(len(dataset) * valid_frac)

(x_train, y_train) = nf.tuplelist2listtuple(dataset[0:train_end_index])
(x_valid, y_valid) = nf.tuplelist2listtuple(dataset[train_end_index + 1 : valid_end_index])
(x_test, y_text) = nf.tuplelist2listtuple(dataset[valid_end_index + 1 : len(dataset)])


# Model layers. Using default from tensorflow tutorial, will experiment with optuna at a later stage

model = keras.saving.load_model("../models/306.keras")

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25)