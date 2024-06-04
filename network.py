import math
# numpy for arrays
import numpy as np
# os for file reading
import os
# nptdms for reading NI output files
import nptdms
# pyplot for plotting vibration signals
import matplotlib.pyplot as plt
# Random for shuffling dataset
import random
# Tensorflow and keras for neural network
import tensorflow as tf

############
# CONSTANTS:
############
INPUT_SIZE = 2048

############
# FUNCTIONS:
############

# get_dataset(dir): reads all tdms files in directory and adds them to an array of tuples containing input and output
def get_dataset(dir):
    # dataset contains tuples of (input_data, output_data)
    dataset = []

    # Get data from each tdms file in directory
    for tdms_filename in os.listdir(dir):
        # Find data array from tdms
        tdms_path = dir + tdms_filename
        tdms_file = nptdms.TdmsFile.read(tdms_path)
        group = tdms_file["Group Name"]
        channel = group["Voltage_0"]
        channel_data = channel[:]

        # Split up channel data into arrays of size INPUT_SIZE
        total_samples = channel_data.size
        index = 0

        while index + INPUT_SIZE < total_samples:
            # FIXME
            # Currently setting all output data to 0, we will decide how to convey this when we develop the physical testing system further
            output_data = 0
            # Input data is the fft of the selected portion of the wave input
            input_data = np.fft.fft(channel_data[(index) : (index + INPUT_SIZE)])
            # Add data to the dataset
            data_tuple = (input_data, output_data)
            dataset.append(data_tuple)
            # Increment index and repeat for next section
            index = index + INPUT_SIZE
    return dataset

# Turns the shuffleable list of tuples to an easier to work with tuple of lists
def tuplelist2listtuple(tuplist):
    list1 = []
    list2 = []

    for item in tuplist:
        list1.append(item[0])
        list2.append(item[1])
    return (list1, list2)


#######
# MAIN:
#######

# Get dataset from tdms files
dataset = get_dataset("./tdms_data/")
# Shuffle dataset
random.shuffle(dataset)

# Partition dataset
train_frac = 0.7
valid_frac = 0.2

train_end_index = math.floor(len(dataset) * train_frac)
valid_end_index = train_end_index + math.floor(len(dataset) * valid_frac)

(x_train, y_train) = tuplelist2listtuple(dataset[0:train_end_index])
(x_valid, y_valid) = tuplelist2listtuple(dataset[train_end_index + 1 : valid_end_index])
(x_test, y_text) = tuplelist2listtuple(dataset[valid_end_index + 1 : len(dataset)])


# Model layers. Using default from tensorflow tutorial, will experiment with optuna at a later stage

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(INPUT_SIZE, 1)),
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanAbsoluteError(
    reduction='sum_over_batch_size',
    name='mean_absolute_error'
)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)