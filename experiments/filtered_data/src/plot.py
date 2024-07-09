import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import nptdms
import re
import ast
import os

def tdms_to_numpy(tdms_path):
    # Find data array from tdms
    tdms_file = nptdms.TdmsFile.read(tdms_path)
    group = tdms_file["Group Name"]
    channel = group["Voltage_0"]
    channel_data = channel[:]
    return channel_data

def plot_log_file(log_path):
    # Plots a .log function created from network_optuna

    ###########################
    ##   Reading .log file   ##
    ###########################

    ## Trial [value] finished with value: [decimal value] and parameters: [new dictionary of parameters]
    trial_format = re.compile(r"Trial (\d+) finished with value: ([\d\.]+) and parameters: ({.*?}).")

    ## Modify this to change log path
    if log_path == None:
        log_path = "src/study.log"

    ## Open log
    file = open(log_path,"r")

    ## Skip first line; this can be modified to make use of the study name
    first_line = file.readline

    ## Create trials array
    trials = []

    ## Parse log file and store trial info
    for line in file:
        curr_trial = trial_format.search(line)              ## Check if line follows format
        if curr_trial:                                      ## If yes...
            trial_num = int(curr_trial.group(1))                ## Gather trial number, finished value, and parameters
            val = float(curr_trial.group(2))
            param = ast.literal_eval(curr_trial.group(3))

            trial_details = {                               ## Group our info, assign tags
                "trial_num" : trial_num,
                "val" : val,
                "param" : param
            }

            trials.append(trial_details)                    ## Append to trials array

    ######################
    ##   Create graph   ##
    ######################

    ## Assign axes
    x_axis = [trial["trial_num"] for trial in trials]
    y_axis = [trial["val"] for trial in trials]

    ## Plotting 
    plt.plot(x_axis, y_axis)
    
    ## Assign labels and title
    plt.xlabel('Trial Number')
    plt.ylabel('Value')
    plt.title('Test')
    
    # Display plot
    plt.grid(True, which="both", ls="--")
    plt.show()

def plot_spectrogram(tdms_path):
    fs = 12800  # Sample rate, may or may not be correct

    array = tdms_to_numpy(tdms_path)

    # Compute the spectrogram
    frequencies, times, Sxx = spectrogram(array, fs)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.show()
    
    return

if __name__ == "__main__":
    while True:
        file_path = input("Enter full path to desired tdms file: ")
        
        if os.path.isfile(file_path):
            print("File path is valid. Generating spectrogram...")
            break
        else:
            print("File path is not valid. Please try again.")
            
    plot_spectrogram(file_path)
    