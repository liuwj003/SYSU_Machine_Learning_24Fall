import numpy as np


def hinge_loss(labels: np.ndarray, samples: np.ndarray, weights: np.ndarray, bias, lamda):
    """
    implementation of hinge loss

    :param labels: one-dim vector, (N, ), and each y in {-1, 1}
    :param samples: matrix, (N, m)
    :param weights: one-dim vector, (m, )
    :param bias: scalar
    :param lamda: regularization strength
    :return: hinge loss(a scalar) result of N samples
    """
    predict = np.dot(weights, samples.T) + bias
    vec_yh = np.multiply(labels, predict)
    ESV_result = np.maximum(0, 1 - vec_yh)
    loss_result = np.sum(ESV_result) / len(labels) + lamda * np.sum(weights ** 2)  # average
    return loss_result


def grad_hinge(labels: np.ndarray, samples: np.ndarray, weights: np.ndarray, bias, lamda):
    """
    Partial derivative of the loss function with respect to w,b

    :param labels: one-dim vector, (N, ), and each y in {-1, 1}
    :param samples: matrix, (N, m)
    :param weights: one-dim vector, (m, )
    :param bias: scalar
    :param lamda: regularization strength
    :return: (grad with respect to w, grad with respect to b)
    """
    predict = np.dot(weights, samples.T) + bias
    compare_with_1_bool = np.multiply(labels, predict) < 1  # return a bool array
    compare_with_1 = compare_with_1_bool.astype(int)  # return a int array
    compare_column = compare_with_1[:, np.newaxis]  # add newaxis for later computing
    labels_column = labels[:, np.newaxis]  # add newaxis for later computing
    matrix_yx = - (samples * labels_column)
    grad_w = 2 * lamda * weights + (np.sum(matrix_yx * compare_column, axis=0) / len(labels))  # average
    grad_b = np.dot(-labels, compare_with_1)
    return grad_w, grad_b


def sigmoid(x):
    """
    x can be scalar or vector, since "/" and 'np.exp' support broadcasting
    """
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(labels: np.ndarray, samples: np.ndarray, weights: np.ndarray, bias, lamda):
    """
    Implementation of cross entropy loss

    :param labels: one-dim vector, (N, ), and each y in {0, 1}
    :param samples: matrix, (N, m)
    :param weights: one-dim vector, (m, )
    :param bias: scalar
    :param lamda: regularization strength
    :return: result of CE loss(a scalar)
    """
    predict = sigmoid(np.dot(weights, samples.T) + bias)
    result_no_regularize = - np.sum(labels * np.log(predict + 1e-9) + (1 - labels) * np.log(1 - predict + 1e-9))
    result = result_no_regularize / len(labels) + lamda * np.sum(weights ** 2)  # average
    return result


def grad_cross_entropy(labels: np.ndarray, samples: np.ndarray, weights: np.ndarray, bias, lamda):
    """

    :param labels: one-dim vector, (N, ), and each y in {0, 1}
    :param samples: matrix, (N, m)
    :param weights: one-dim vector, (m, )
    :param bias: scalar
    :param lamda: regularization strength
    :return: grad of CE loss with respect to w, b
    """
    predict = sigmoid(np.dot(weights, samples.T) + bias)
    grad_w = np.dot(samples.T, predict - labels) / len(labels) + 2 * lamda * weights  # average
    grad_b = np.sum(predict - labels) / len(labels)  # average
    return grad_w, grad_b
