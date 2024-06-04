# numpy for arrays
import numpy as np
# os for file reading
import os
# nptdms for reading NI output files
import nptdms
# pyplot for plotting vibration signals
import matplotlib.pyplot as plt
# Tensorflow and keras for neural network
#import tensorflow as tf

# CONSTANTS:
INPUT_SIZE = 2048

# FUNCTIONS:

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


# MAIN:

def main():
    # Get dataset from tdms files
    dataset = get_dataset("./tdms_data/")
    print(dataset)

main()