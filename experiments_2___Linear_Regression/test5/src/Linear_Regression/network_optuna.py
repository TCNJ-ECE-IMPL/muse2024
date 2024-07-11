import math
import numpy as np
import os
import shutil
import nptdms
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import optuna
import logging
import network_functions as nf
import pathlib

# Tests different model structures and determines the most effective one. This is then saved to "checkpoints"

############
# CONSTANTS:
############

INPUT_SIZE = 1024
USE_FFT = True
EPOCHS = 15
N_TRIALS = 500

train_ds = None
valid_ds = None

############
# FUNCTIONS:
############

def create_model(trial):
    # This portion of code is directly from 2CC
    n_layers = trial.suggest_int("n_layers", 1, 10)
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
    # This is where the 2 class layer is changed to 1 output
    model.add(
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    )
    return model

# Copied directly from 2CC
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
    train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))

    model = create_model(trial)
    optimizer = create_optimizer(trial) 

    with tf.device("/gpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, "train")

        mse = learn(model, optimizer, valid_ds, "eval")

    model_save_filename = str(trial.number) + ".keras"
    model_save_path = os.path.join(nf.getPath("../models/temp"), model_save_filename)
    model.save(model_save_path)

    # Return last validation accuracy.
    return mse.result()

# Modified from 2CC
def learn(model, optimizer, dataset, mode="eval"):
    mse = tf.keras.losses.MeanSquaredError()

    for batch, (features, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(features, training=(mode == "train"))
            loss_value = mse(labels, predictions) 
            if mode == "eval":
                mse.update_state(labels, predictions)
            else:
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if mode == "eval":
        return mse.result()

if __name__ == "__main__":

    
    # Get dataset

    train_ds, valid_ds = nf.get_dataset(nf.getPath("../tdms_data/"))

    # Set up logging within experiment directory
    temppath = pathlib.Path().resolve()
    logpath = os.path.join(temppath, nf.getPath("study.log"))

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(logpath, mode="w"))

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    # Set up directory for saving models
    os.mkdir(nf.getPath("../models/temp"))

    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    # Find and preserve best model

    trial = study.best_trial
    best_model_filename = str(trial.number) + ".keras"
    old_best_path = os.path.join(nf.getPath("../models/temp"), best_model_filename)
    new_best_path = os.path.join(nf.getPath("../models"), "optuna_best.keras")
    os.replace(old_best_path, new_best_path)
    #shutil.rmtree(nf.getPath("../models/temp"))

    # Report best model

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))