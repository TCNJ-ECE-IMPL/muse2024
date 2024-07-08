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

# Creates a new model using user defined parameters and trains it

############
# CONSTANTS:
############
INPUT_SIZE = 1024
USE_FFT = True
NUM_EPOCHS = 50

#######
# MAIN:
#######

# Get dataset from tdms files
train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))


l2 = tf.keras.regularizers.L2(
    l2=0.001
)

model = keras.Sequential(
    [
        #layers.InputLayer(shape=(1024)),
        #layers.Conv1D(32, [11], activation="relu", name="conv1", data_format="channels_first"),
        #layers.Flatten(data_format="channels_first"),
        layers.Dense(86, activation="relu", name="fc1"),
        layers.Dropout(0.2),
        layers.Dense(40, activation="relu", kernel_regularizer=l2, name="fc2"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(2, name="output")
    ]
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.002375367385948983,
    weight_decay=1.0325812267590434e-08
)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Prepare saving of checkpoints

checkpoint_path = nf.getPath("../checkpoints/checkpoints_3.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Run the model

history = model.fit(
    x=train_ds, 
    validation_data=valid_ds, 
    epochs=NUM_EPOCHS,
    callbacks=[cp_callback]
)

nf.plotTraining(history)