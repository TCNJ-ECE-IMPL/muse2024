import math
import numpy as np
import os
import nptdms
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import network_functions as nf
import keras
from plot import plotTraining

# Picks up an optuna model and trains it further

############
# CONSTANTS:
############
NUM_EPOCHS = 50

#######
# MAIN:
#######

# Configure GPU

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])

# Get dataset from tdms files
train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))


# Model layers. Using default from tensorflow tutorial, will experiment with optuna at a later stage

model = keras.saving.load_model(nf.getPath("../models/5.keras"))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Load weights from previous checkpoint

old_checkpoint_path = nf.getPath("../checkpoints/18.weights.h5")
model.load_weights(old_checkpoint_path)

# Prepare saving of checkpoints

checkpoint_path = nf.getPath("../checkpoints/5.weights.h5")

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Run the model

history = model.fit(
    x=train_ds, 
    validation_data=valid_ds, 
    epochs=NUM_EPOCHS,
    callbacks=[cp_callback]
)

plotTraining(history)