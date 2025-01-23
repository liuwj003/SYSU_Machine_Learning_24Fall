import pickle
import numpy as np
import torch


def load_data(dir): # dir = "E:/2024-1/ML/Assignment2/data"
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X_train.append(dict[b'data'])
            Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)

    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X_test = dict[b'data']
        Y_test = dict[b'labels']

    return X_train, Y_train, X_test, Y_test


def standardization(samples):
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    std_adjust = np.where(std == 0, 1, std)
    samples = (samples - mean) / std_adjust
    return samples


def reshape(model, X_train, Y_train, X_test, Y_test):
    """
    :param model: Softmax or CNN or MLP
    :param X_train: Training Samples
    :param Y_train: Training Labels
    :param X_test: Test Samples
    :param Y_test: Test Labels
    :return: Tensor Type, reshaped for specific model
    """
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.int64)
    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.int64)

    if model in {'CNN', 'CNN2', 'CNN3', 'CNN4', 'CNN5'}:
        X_train_tensor = X_train_tensor.view(-1, 3, 32, 32)
        X_test_tensor = X_test_tensor.view(-1, 3, 32, 32)
    else:  # for MLP or Softmax
        X_train_tensor = X_train_tensor.view(-1, 3072)
        X_test_tensor = X_test_tensor.view(-1, 3072)

    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor






