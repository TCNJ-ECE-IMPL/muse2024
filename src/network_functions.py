import math
import numpy as np
import nptdms
import tensorflow as tf
import os
import random

INPUT_SIZE = 1024
USE_FFT = True

# reads all tdms files in directory and adds them to an array of tuples containing input and output
def get_base_dataset(dir):
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

        while index + INPUT_SIZE * 2 < total_samples:
            # FIXME
            # Currently setting all output data to 0, we will decide how to convey this when we develop the physical testing system further
            if tdms_filename == "exp1.tdms":
                output_data = 0.0
            else:
                output_data = 1.0

            # Input data is the fft of the selected portion of the wave input
            if USE_FFT:
                input_data_unfiltered = np.absolute(np.fft.fft(channel_data[(index) : (index + INPUT_SIZE * 2)]))
            else:
                input_data_unfiltered = channel_data[(index) : (index + INPUT_SIZE * 2)]
            
            input_data = filterHighest(input_data_unfiltered)

            # Add data to the dataset
            data_tuple = (input_data, output_data)
            dataset.append(data_tuple)
            
            # Increment index and repeat for next section
            index = index + INPUT_SIZE * 2
    
    return dataset

# Gets the dataset and formats to 
def get_dataset(dir):
    # Get dataset from tdms files
    dataset = get_base_dataset(dir)
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

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    return train_ds, valid_ds

# Turns the shuffleable list of tuples to an easier to work with tuple of lists
def tuplelist2listtuple(tuplist):
    num_items = len(tuplist)
    list1 = np.empty((num_items, INPUT_SIZE))
    list2 = np.empty((num_items, 1))

    for i in range(len(tuplist)):
        item = tuplist[i]
        l1a = np.array([item[0]])
        list1[i] = l1a
        l2a = np.array([item[1]])
        list2[i] = l2a
    return (list1, list2)

# Filters out highest value moduli
def filterHighest(input_data_unfiltered):
    # Filter out highest 1024
    median = np.median(input_data_unfiltered)
    input_data = np.array([])
    for item in input_data_unfiltered:
        if item <= median:
            input_data = np.append(input_data, item)

    # Ensure duplicates of median do not push over the edge
    while input_data.size > INPUT_SIZE:
        for i in range(input_data.size):
            if input_data[i] == median:
                input_data = np.delete(input_data, i)
                break
    return input_data
