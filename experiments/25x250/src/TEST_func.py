import math
import numpy as np
import nptdms
import tensorflow as tf
import os
import random
import csv

INPUT_SIZE = 1024
USE_FFT = True
CLASS_NAMES = ['Secure', 'Compromised']

# reads all tdms files in directory and adds them to an array of tuples containing input and output
def get_base_dataset(dir):
    x = np.empty([1, 1, INPUT_SIZE], dtype=float)
    y = np.empty([1], dtype=int)


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
            input_data_unfiltered = np.fft.fft(channel_data[(index) : (index + INPUT_SIZE * 2)])
            input_data_unfiltered_modulus = np.absolute(input_data_unfiltered)
            sorted = np.sort(input_data_unfiltered)

            bwomp = sorted[1320]
            fuckassnum1 = np.where(input_data_unfiltered == (0.0004113074679999995+0j))
            fuckassnum2 = np.where(input_data_unfiltered == (0.04309775239000009+0j))

            
            
            x=1
    






#####################################################################################################################

# Takes base dataset arrays, shuffles, and splits into training and validation. returns as tensorflow datasets
def get_dataset(dir):
    (x, y) = get_base_dataset(dir)

    # Finds absolute path from relative with respect to executing file
def getPath(rel):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)
