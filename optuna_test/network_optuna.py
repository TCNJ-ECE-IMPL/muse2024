import math
import numpy as np
import os
import nptdms
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import optuna

############
# CONSTANTS:
############
INPUT_SIZE = 2048
USE_FFT = False
EPOCHS = 5

############
# FUNCTIONS:
############

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

        while index + INPUT_SIZE < total_samples:
            # FIXME
            # Currently setting all output data to 0, we will decide how to convey this when we develop the physical testing system further
            output_data = np.array([0])

            # Input data is the fft of the selected portion of the wave input
            if USE_FFT:
                input_data_unfiltered = np.absolute(np.fft.fft(channel_data[(index) : (index + INPUT_SIZE)]))
            else:
                input_data_unfiltered = channel_data[(index) : (index + INPUT_SIZE)]
            
            # Filter out highest 1024
            median = np.median(input_data_unfiltered)
            input_data = np.array([])
            for item in input_data_unfiltered:
                if item <= median:
                    input_data = np.append(input_data, item)

            # Ensure duplicates of median do not push over the edge
            while input_data.size > INPUT_SIZE/2:
                for i in range(input_data.size):
                    if input_data[i] == median:
                        input_data = np.delete(input_data, i)
                        break

            # Add data to the dataset
            data_tuple = (input_data, output_data)
            dataset.append(data_tuple)
            
            # Increment index and repeat for next section
            index = index + INPUT_SIZE
    
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

    for item in tuplist:
        list1 = np.append(list1, [item[0]], axis=0)
        list2 = np.append(list2, [item[1]], axis=0)
    return (list1, list2)

# Modified from optuna example on GitHub
def learn(model, optimizer, dataset, mode="eval"):
    accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

    for batch, (images, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images, training=(mode == "train"))
            loss_value = tf.reduce_mean(tf.losses.mean_squared_error(labels, logits))
            if mode == "eval":
                accuracy(
                    tf.argmax(logits, axis=1, output_type=tf.int64), tf.cast(labels, tf.int64)
                )
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == "eval":
        return accuracy

# Modified from optuna example on GitHub
def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
    model.add(
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    )
    return model

# Modified from optuna example on GitHub
def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["weight_decay"] = trial.suggest_float("rmsprop_weight_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

# Modified from optuna example on GitHub
def objective(trial):
    # Get dataset from tdms files
    train_ds, valid_ds = get_dataset("./tdms_data/")

    model = create_model(trial)
    optimizer = create_optimizer(trial) 

    with tf.device("/cpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, "train")

        accuracy = learn(model, optimizer, valid_ds, "eval")

    # Return last validation accuracy.
    return accuracy.result()

#######
# MAIN:
#######

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))