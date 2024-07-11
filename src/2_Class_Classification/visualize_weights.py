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


# Load model

model = keras.saving.load_model(nf.getPath("../models/optuna_best.keras"))

checkpoint_path = nf.getPath("../checkpoints/checkpoints_1.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# Obtain an input from test dataset

dataset_path = nf.getPath("../datasets/")

test_ds = tf.data.Dataset.load(os.path.join(dataset_path, "test"))

test_batch = get_ds_element(test_ds, 0)

X = test_batch[0][0]
sample_out = test_batch[1][0]

# Get the weights and biases of the first dense layer
W1 = model.layers[0].get_weights()

# Get the weights and biases of the second dense layer
W2 = model.layers[1].get_weights()

Y = keras.activations.relu(W1 * X)

W1_eff = Y / X

Z = tf.nn.softmax(W2 * W1_eff * X)

print("hi")