import pandas as pd
import numpy as np


def extract_data(file_name: str):
    data = pd.read_csv(file_name)
    # data_train = pd.read_csv('mnist_01_train.csv')
    labels = data['label'].values
    # store samples' attributes matrix, each row refers to one sample
    samples = data.iloc[:, 1:].values
    return labels, samples


def standardization(samples):
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    std_adjust = np.where(std == 0, 1, std)
    samples = (samples - mean) / std_adjust
    return samples


def min_max_normalization(samples):
    col_min = np.min(samples, axis=0)
    col_max = np.max(samples, axis=0)
    difference_adjust = np.where((col_max - col_min) == 0, 1, (col_max - col_min))
    samples = (samples - col_min) / difference_adjust
    return samples


def data_shuffle(samples, labels):
    """return shuffled_samples, shuffled_labels"""
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    shuffled_samples = samples[indices]
    shuffled_labels = labels[indices]
    return shuffled_samples, shuffled_labels
