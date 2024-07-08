import network_functions as nf
import numpy as np
import nptdms
import os
import math
import tensorflow as tf

def tdms_to_numpy(tdms_path):
    # Find data array from tdms
    tdms_file = nptdms.TdmsFile.read(tdms_path)
    group = tdms_file["Group Name"]
    channel = group["Voltage_0"]
    channel_data = channel[:]
    return channel_data

INPUT_SIZE = nf.INPUT_SIZE
USE_FFT = True


# Initialize input and output arrays
x = np.empty([1, 1, INPUT_SIZE], dtype=float)
y = np.empty([1], dtype=int)

tdms_dir = nf.getPath("../tdms_data/")
# Get data from each tdms file in directory
for tdms_filename in os.listdir(tdms_dir):
    tdms_path = tdms_dir + tdms_filename
    channel_data = tdms_to_numpy(tdms_path)

    # Split up channel data into arrays of size INPUT_SIZE
    total_samples = channel_data.size
    index = 0

    while index + INPUT_SIZE * 2 < total_samples:
        if "secure" in tdms_filename:
            output_data = 0
        else:
            output_data = 1

        # Input data is the fft of the selected portion of the wave input
        if USE_FFT:
            input_data = np.absolute(np.fft.fft(channel_data[(index) : (index + INPUT_SIZE * 2)]))[0:INPUT_SIZE]
        else:
            input_data = channel_data[(index) : (index + INPUT_SIZE * 2)]
        
        input_data = np.array([input_data])

        # Add data to the dataset
        x = np.append(x, [input_data], axis=0)
        y = np.append(y, [output_data], axis=0)

        # Increment index and repeat for next section
        index = index + INPUT_SIZE * 2


# Remove blank entries from initilization
x = np.delete(x, 0, axis=0)
y = np.delete(y, 0, axis=0)

# Shuffle
(x, y) = nf.unison_shuffled_copies(x, y)

# Calculate partition sizes
train_frac = 0.7
valid_frac = 0.2

train_end_index = math.floor(len(x) * train_frac)
valid_end_index = train_end_index + math.floor(len(x) * valid_frac)

# Partition datasets
x_train = x[0:train_end_index]
y_train = y[0:train_end_index]
x_valid = x[train_end_index + 1 : valid_end_index]
y_valid = y[train_end_index + 1 : valid_end_index]
x_test = x[valid_end_index + 1 :]
y_test = y[valid_end_index + 1 :]

# Create tensorflow dataset objects
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.batch(batch_size=25)

valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_ds = valid_ds.batch(batch_size=25)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(batch_size=25)

# Save datasets to files
dataset_path = nf.getPath("../datasets/")
tf.data.Dataset.save(train_ds, os.path.join(dataset_path, "train"))
tf.data.Dataset.save(valid_ds, os.path.join(dataset_path, "valid"))
tf.data.Dataset.save(test_ds,  os.path.join(dataset_path,  "test"))
