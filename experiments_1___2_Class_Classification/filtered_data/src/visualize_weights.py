import tensorflow as tf
import network_functions as nf
import numpy as np
import keras
import os
import matplotlib.pyplot as plt

# Used to get an individual batch from dataset object
def get_ds_element(ds, number):
    # Case for invalid input
    if number < 0:
        return None
    
    # Dataset object must be accessed by an iterator. *sigh* 
    for element in ds:
        if number == 0:
            return element
        number = number - 1
    return None

# Solves for effective weights
def effective_weight(Y, W):
    # Eliminate unnecessary dimensions
    Y = Y[0]

    for y_index in range(Y.size):
        if Y[y_index] == 0.0:
            # If Y is cut off by ReLU, set effective weight to 0
            for w_element in W:
                w_element[y_index] = 0
    return W

# Plots a numpy array. A little ugly but almost does the trick
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

# Load model

model = keras.saving.load_model(nf.getPath("../models/80p2n.keras"))

checkpoint_path = nf.getPath("../checkpoints/checkpoints.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

# Obtain an input from test dataset

dataset_path = nf.getPath("../datasets/")

test_ds = tf.data.Dataset.load(os.path.join(dataset_path, "test"))

# Displays graphs for 5 testing data inputs to display differences between classes
for i in range(5):

    # Obtain input from test dataset
    test_batch = get_ds_element(test_ds, 0)

    X = test_batch[0][i].numpy()
    result = test_batch[1][i].numpy()

    # Obtain weights from layers
    W1_raw = model.layers[1].get_weights()
    W2_raw = model.layers[2].get_weights()

    # Convert weights to a more workable format
    W1 = tf.cast(W1_raw[0], tf.float64).numpy()
    W2 = tf.cast(W2_raw[0], tf.float64).numpy()

    # Solve for Y
    temp = tf.linalg.matmul(X, W1)
    Y = keras.activations.relu(temp).numpy()

    # Use Y to find effective weights
    W1_eff = effective_weight(Y, W1)

    # Plot
    vis_weights = tf.linalg.matmul(W1_eff, W2).numpy()

    visualize_array(vis_weights, "Result = " + str(result))
