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

model = keras.saving.load_model(nf.getPath("/home/aldridf1/muse2024/muse2024/AA_Notable Results/OneLayer_99perc2neur/80p2n.keras"))

checkpoint_path = nf.getPath("/home/aldridf1/muse2024/muse2024/AA_Notable Results/OneLayer_99perc2neur/checkpoints.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# Obtain an input from test dataset

dataset_path = nf.getPath("../datasets/")

test_ds = tf.data.Dataset.load(os.path.join(dataset_path, "test"))

test_batch = get_ds_element(test_ds, 0)

X = test_batch[0][0]
sample_out = test_batch[1][0]

L1_weights = model.layers[1].get_weights()

W1 = tf.cast(L1_weights[0], tf.float64)
W2 = tf.cast(L1_weights[1], tf.float64)

temp = tf.linalg.matmul(X, W1)
Y = keras.activations.relu(temp)

W1_eff = (Y / X)

temp = tf.linalg.matmul(W2, W1_eff)
temp = tf.linalg.matmul(temp, X)

Z = tf.nn.softmax(temp)

print("hi")