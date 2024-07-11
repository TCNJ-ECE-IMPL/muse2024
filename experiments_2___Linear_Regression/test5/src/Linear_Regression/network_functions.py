import math
import numpy as np
import nptdms
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt

INPUT_SIZE = 1024
CLASS_NAMES = ['Secure', 'Compromised']
DATA_PATH = "../datasets/"

# Shuffles two numpy arrays the same way - used to shuffle x and y together
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_dataset(dir):
    # Load datasets from files
    dataset_path = getPath(DATA_PATH)
    train_ds = tf.data.Dataset.load(os.path.join(dataset_path, "train"))
    valid_ds = tf.data.Dataset.load(os.path.join(dataset_path, "valid"))

    return train_ds, valid_ds

def getPath(rel):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("../", rel))
