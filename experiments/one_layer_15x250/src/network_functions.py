import math
import numpy as np
import nptdms
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

# A set of predefined functions

INPUT_SIZE = 1024
CLASS_NAMES = ['Secure', 'Compromised']


# Shuffles two numpy arrays the same way - used to shuffle x and y together
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Shorthand to get training and validation datasets from files
def get_dataset(dir):
    # Load datasets from files
    dataset_path = getPath("../datasets/")
    train_ds = tf.data.Dataset.load(os.path.join(dataset_path, "train"))
    valid_ds = tf.data.Dataset.load(os.path.join(dataset_path, "valid"))

    return train_ds, valid_ds

# Finds absolute path from relative with respect to executing file
def getPath(rel):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)

