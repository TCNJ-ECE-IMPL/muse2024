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
INPUT_SIZE = 1024
USE_FFT = True
NUM_EPOCHS = 5

#######
# MAIN:
#######

# Get dataset from tdms files
train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))


# Model layers. Using default from tensorflow tutorial, will experiment with optuna at a later stage

model = keras.saving.load_model(nf.getPath("../models/optuna_best.keras"))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Prepare saving of checkpoints

checkpoint_path = nf.getPath("../checkpoints/best_start.weights.h5")

best_acc = 0.5

# Run the model

while True:

    model.fit(
        x=train_ds, 
        validation_data=valid_ds, 
        epochs=NUM_EPOCHS
    )
    #print(model.get_metrics_result())
    acc = model.get_metrics_result()["accuracy"]

    if acc > best_acc:
        best_acc = acc
        model.save_weights(checkpoint_path)
    
    print("Best validation accuracy: " + str(best_acc))

