import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import network_functions as nf
import seaborn as sns
import os

######## CONSTANTS ########
MODEL_PATH = "../../models/5.keras"
CP_PATH = "../../checkpoints/5.weights.h5"
TDMS_PATH = "../../tdms_data/"

INPUT_SIZE = nf.INPUT_SIZE
USE_FFT = True
NAVG = 16



def find_misclassifications(tdms_files):
    output_arr = []

    # Load the model
    model = tf.keras.models.load_model(nf.getPath(MODEL_PATH))

    # Restore the latest checkpoint
    latest_checkpoint = nf.getPath(CP_PATH)
    model.load_weights(latest_checkpoint)

    # Load tdms files

    tdms_dir = nf.getPath(TDMS_PATH)

    for tdms_filename in tdms_files:
        tdms_path = tdms_dir + tdms_filename
        channel_data = nf.tdms_to_numpy(tdms_path)

        # Split up channel data into arrays of size INPUT_SIZE
        
        total_samples = channel_data.size
        index = 0
        x = np.empty([1, 1, INPUT_SIZE], dtype=float)
        y = np.empty([1], dtype=int)
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

            # Add data to the dataset
            x = np.append(x, [input_data], axis=0)
            y = np.append(y, [output_data], axis=0)

            # Increment index and repeat for next section
            index = index + INPUT_SIZE * 2
        
        # Remove blank entries from initilization
        x = np.delete(x, 0, axis=0)
        y = np.delete(y, 0, axis=0)

        # Create tensorflow dataset objects
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.batch(batch_size=25)

        # Convert dataset to a list to inspect structure
        data_list = list(ds.as_numpy_iterator())

        # Extract data and labels based on inspected structure
        inputs = np.array([item[0][0] for item in data_list])  
        labels = np.array([item[1][0] for item in data_list]) 

        # Make predictions
        predictions = model.predict(inputs)
        predicted_classes = np.argmax(predictions, axis=1)

        # Make comparisons between real and predicted
        index = 0
        correct = 0
        mislabel_secure = 0
        mislabel_compromised = 0
        while index < predicted_classes.size:
            if predicted_classes[index] == labels[index]:
                correct += 1
            elif predicted_classes[index] == 0:
                mislabel_secure += 1
            elif predicted_classes[index] == 1:
                mislabel_compromised += 1
            else:
                correct = 9999
            index += 1
        
        mislabel_prop = mislabel_secure + mislabel_compromised / (mislabel_secure + mislabel_compromised + correct)

        # Append results to output
        trial_details = {                         
                "file" : tdms_filename,
                "correct" : correct,
                "mislabel_secure" :  mislabel_secure,
                "mislabel_compromised" : mislabel_compromised,
                "mislabel_prop" : mislabel_prop
            }
        
        output_arr.append(trial_details)

    return output_arr




def generate_confusion_mtx():
    # Load the model
    model = tf.keras.models.load_model(nf.getPath(MODEL_PATH))

    # Load the test data
    test_data = tf.data.Dataset.load(nf.getPath("../datasets/train"))

    # Convert dataset to a list to inspect structure
    test_data_list = list(test_data.as_numpy_iterator())

    # Extract data and labels based on inspected structure
    test_inputs = np.array([item[0][0] for item in test_data_list])  
    test_labels = np.array([item[1][0] for item in test_data_list])  

    # Restore the latest checkpoint
    latest_checkpoint = nf.getPath(CP_PATH)
    model.load_weights(latest_checkpoint)

    # Make predictions
    predictions = model.predict(test_inputs)
    predicted_classes = np.argmax(predictions, axis=1)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["secure", "compromised"], yticklabels=["secure", "compromised"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    return

if __name__ == "__main__":
    tdms_files = []
    tdms_dir = nf.getPath(TDMS_PATH)
    for tdms_filename in os.listdir(tdms_dir):
        tdms_files.append(tdms_filename)
    
    output = find_misclassifications(tdms_files)

    for trial_details in output:
        print("File:", trial_details["file"])
        print("Correct:", trial_details["correct"])
        print("Mislabeled Secure:", trial_details["mislabel_secure"])
        print("Mislabeled Compromised:", trial_details["mislabel_compromised"])
        print()  

    print("Top Files Misclassified:")

    output.sort(key=lambda x: x["mislabel_prop"], reverse=True)

    print(output[0])
    print(output[1])
    print(output[2])    