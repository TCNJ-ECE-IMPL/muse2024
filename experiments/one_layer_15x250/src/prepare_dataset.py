import network_functions as nf
import numpy as np
import nptdms
import os
import math
import tensorflow as tf
import time
import matplotlib.pyplot as plt

def tdms_to_numpy(tdms_path):
    # Find data array from tdms
    tdms_file = nptdms.TdmsFile.read(tdms_path)
    group = tdms_file["Group Name"]
    channel = group["Voltage_0"]
    channel_data = channel[:]
    return channel_data

INPUT_SIZE = nf.INPUT_SIZE
USE_FFT = True
NAVG = 16

# creating initial data values
# of x and y
x = np.linspace(0, 1024, 1024)
y = 0.8*np.absolute(np.sin(x))

# to run GUI event loop
plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)

# setting title
plt.title("Geeks For Geeks", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

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
    print(tdms_filename)
    while index + INPUT_SIZE * 2 * NAVG < total_samples:
        if "secure" in tdms_filename:
            output_data = 0
        else:
            output_data = 1

        for avg_offset in range(NAVG):
            #print("fn=%s, index=%d, class=%d" % (tdms_filename, index, output_data))
            # Input data is the fft of the selected portion of the wave input
            if USE_FFT:
                this_data = np.absolute(np.fft.fft(channel_data[(index+INPUT_SIZE*avg_offset) : (index+INPUT_SIZE*avg_offset + INPUT_SIZE * 2)]))[0:INPUT_SIZE]
            else:
                this_data = channel_data[(index) : (index + INPUT_SIZE * 2)]
            
            this_data[0]=0

            if (avg_offset == 0):
                input_data = np.array([this_data])
            else:
                input_data += np.array([this_data])

        # updating data values
        #line1.set_xdata(range(INPUT_SIZE))
        #line1.set_ydata(input_data)

        # drawing updated values
        #figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        #figure.canvas.flush_events()

        #time.sleep(0.01667)

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
