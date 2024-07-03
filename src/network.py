import math
import numpy as np
import os
import nptdms
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import network_functions as nf
import keras
from keras import layers

############
# CONSTANTS:
############
INPUT_SIZE = 1024
USE_FFT = True
NUM_EPOCHS = 500

#######
# MAIN:
#######

# Get dataset from tdms files
train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))


# Model layers. Using default from tensorflow tutorial, will experiment with optuna at a later stage

model = keras.Sequential(
    [
        layers.Conv1D(32, [11], activation="relu", name="conv1", data_format="channels_first"),
        layers.Flatten(data_format="channels_first"),
        layers.Dense(7, activation="relu", name="fc1"),
        layers.Dense(12, activation="relu", name="fc2"),
        layers.Dense(71, activation="relu", name="fc3"),
        layers.Dense(122, activation="relu", name="fc4"),
        layers.Dense(7, activation="relu", name="fc5"),
        layers.Dense(6, activation="relu", name="fc6"),
        layers.Dense(126, activation="relu", name="fc7"),
        layers.Dense(5, activation="relu", name="fc8"),
        layers.Dense(2, activation="softmax", name="output")
    ]
)

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=1.07600032481164e-05,
    weight_decay=0.9826945740093362,
    momentum=1.0331352319816896e-05
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Prepare saving of checkpoints

checkpoint_path = nf.getPath("../checkpoints/checkpoints.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Run the model

model.fit(
    x=train_ds, 
    validation_data=valid_ds, 
    epochs=NUM_EPOCHS,
    callbacks=[cp_callback]
)
