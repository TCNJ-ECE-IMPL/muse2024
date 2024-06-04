# numpy for arrays
import numpy as np
# nptdms for reading NI output files
import nptdms
# pyplot for plotting vibration signals
import matplotlib.pyplot as plt
# Tensorflow and keras for neural network
import tensorflow as tf
import keras

# Extract array data from NI tdms output file
tdms_file = nptdms.TdmsFile.read("./test.tdms")
group = tdms_file["Group Name"]
channel = group["Voltage_0"]
channel_data = channel[:]

# Create array for y axis for plotting purposes
x_axis = np.array(range(channel_data.size))

# Plot vibration signal
plt.plot(x_axis, channel_data)
plt.show()

