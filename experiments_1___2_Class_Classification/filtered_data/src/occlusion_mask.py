import tensorflow as tf
import network_functions as nf
import numpy as np
import keras
import os

def get_ds_element(ds, number):
    if number < 0:
        return None
    for element in ds:
        if number == 0:
            return element
        number = number - 1
    return None

def getResultFromLogits(logits):
    highest = logits[0]
    highest_index = 0

    for index in range(logits):
        if logits[index] > highest:
            highest_index = index
            highest = logits[index]
    return highest_index


# Load model

model = keras.saving.load_model(nf.getPath("../models/optuna_best.keras"))

checkpoint_path = nf.getPath("../checkpoints/checkpoints.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# Obtain an input from test dataset

dataset_path = nf.getPath("../datasets/")

test_ds = tf.data.Dataset.load(os.path.join(dataset_path, "test"))

test_batch = get_ds_element(test_ds, 0)

sample_x = np.array([test_batch[0][0]])
sample_y = test_batch[1][0]

pred_y = getResultFromLogits(model.predict(sample_x)[0])

print(str(pred_y) + " vs " + str(sample_y))