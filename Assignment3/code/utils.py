import numpy as np
from scipy.optimize import linear_sum_assignment


def distance(point_a, point_b):
    """
    compute distance^2 between 2 points
    :param point_a: one sample with 784 dimensions
    :param point_b: another sample with 784 dimensions
    :return: their squared Euclidean distance
    """
    subtract = point_a - point_b
    result = np.sum(subtract ** 2)
    return result


def furthest(point, samples):
    """
    find the furthest point away from this "point"
    :param point:
    :param samples:
    :return: i
    """
    dist = 0
    furthest_point = -1
    for i in range(samples.shape[0]):
        another_point = np.squeeze(samples[i, :])
        curr_dist = distance(point, another_point)
        if curr_dist > dist:
            dist = curr_dist
            furthest_point = i

    return furthest_point


def far(point, samples, k):
    """
    find top-k points furthest away from this "point"
    :param point:
    :param samples:
    :return: k far points
    """
    dist = np.zeros(k)
    far_points = np.full(k, -1, dtype=int)
    for i in range(samples.shape[0]):
        another_point = np.squeeze(samples[i, :])
        curr_dist = distance(point, another_point)
        if curr_dist > np.min(dist):
            min_index = np.argmin(dist)
            dist[min_index] = curr_dist
            far_points[min_index] = i

    return far_points


def calculate_acc(labels, predicts):
    """
    Calculate acc
    :return: ACC
    """
    labels = labels.astype(int)
    predicts = predicts.astype(int)

    # Determine the number of clusters
    n_clusters = int(max(predicts.max(), labels.max())) + 1

    # Create a cost matrix for label matching
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    for i in range(len(labels)):
        cost_matrix[predicts[i], labels[i]] += 1

    # Find the optimal mapping using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Calculate ACC
    acc = cost_matrix[row_ind, col_ind].sum() / len(labels)
    return acc

