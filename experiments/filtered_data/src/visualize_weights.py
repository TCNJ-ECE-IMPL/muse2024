import tensorflow as tf
import network_functions as nf
import numpy as np
import keras
import os
import matplotlib.pyplot as plt

def get_ds_element(ds, number):
    if number < 0:
        return None
    for element in ds:
        if number == 0:
            return element
        number = number - 1
    return None

def effective_weight(Y, W):
    Y = Y[0]
    for y_index in range(Y.size):
        if Y[y_index] == 0.0:
            for w_element in W:
                w_element[y_index] = 0
    return W

def visualize_array(array, title):
    if not isinstance(array, np.ndarray):
        raise ValueError("The input must be a numpy array.")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(array, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def getResultFromLogits(logits):
    highest = logits[0]
    highest_index = 0

    for index in range(logits):
        if logits[index] > highest:
            highest_index = index
            highest = logits[index]
    return highest_index

# Load model

model = keras.saving.load_model(nf.getPath("../models/80p2n.keras"))

checkpoint_path = nf.getPath("../checkpoints/checkpoints.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# Obtain an input from test dataset

dataset_path = nf.getPath("../datasets/")

test_ds = tf.data.Dataset.load(os.path.join(dataset_path, "test"))

for i in range(15):
    test_batch = get_ds_element(test_ds, 0)

    X = test_batch[0][i].numpy()
    result = test_batch[1][i].numpy()

    W1_raw = model.layers[1].get_weights()
    W2_raw = model.layers[2].get_weights()

    W1 = tf.cast(W1_raw[0], tf.float64).numpy()
    W2 = tf.cast(W2_raw[0], tf.float64).numpy()

    temp = tf.linalg.matmul(X, W1)
    Y = keras.activations.relu(temp).numpy()

    W1_eff = effective_weight(Y, W1)

    vis_weights = tf.linalg.matmul(W1_eff, W2).numpy()

    visualize_array(W1_eff, str(result))
