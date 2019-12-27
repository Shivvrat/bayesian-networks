import numpy as np
import pandas as pd


def import_data(dataset_name):
    train_dataset = np.loadtxt("data/" + dataset_name + ".ts.data", delimiter=',')
    test_dataset = np.loadtxt("data/" + dataset_name + ".test.data", delimiter=',')
    valid_dataset = np.loadtxt("data/" + dataset_name + ".valid.data", delimiter=',')
    return train_dataset, test_dataset, valid_dataset
