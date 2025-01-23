import numpy as np
from utils import distance, far, furthest
from tqdm import tqdm

class KMeans:
    def __init__(self, init_method, cluster_num, samples, iters):
        """

        :param init_method: random or dist-based
        :param cluster_num: 10 in this lab
        :param samples: 60000 samples
        :param self.center: clustering center
        :param self.gamma: 60000 rows, each row a clustering one-hot_k for sample_n
        """
        self.init_method = init_method
        self.cluster_num = cluster_num
        self.samples = samples
        self.center = samples[:10, :]
        self.samples_num = samples.shape[0]
        self.gamma = np.zeros((self.samples_num, cluster_num))
        self.iters = iters

    def init_center(self):
        """
        initialize clustering center with random or random+dist-based method
        :return:
        """
        if self.init_method == "random":
            random_indices = np.random.choice(self.samples_num, size=self.cluster_num, replace=False)
            self.center = self.samples[random_indices, :]

        elif self.init_method == "dist-based":
            random_index = np.random.choice(self.samples_num, size=1, replace=False)
            self.center[0, :] = np.squeeze(self.samples[random_index, :])

            # choose next clustering center based on distance
            for i in range(1, 10):
                last_center = self.center[i-1, :]
                k = 100  # we choose top-k far points
                far_k_points = far(last_center, self.samples, k)
                choose = np.random.choice(far_k_points)
                self.center[i, :] = np.squeeze(self.samples[choose, :])

        else:
            raise ValueError("Invalid center-init-method")

    def update_cluster(self):
        """
        update cluster-matrix {gamma_nk}
        :return:
        """
        for i in range(self.samples_num):
            sample_i = self.samples[i]
            distances = np.linalg.norm(self.center - sample_i, axis=1)
            k = np.argmin(distances)
            # One-Hot Code
            self.gamma[i, :] = 0  # Remember to set ZERO !
            self.gamma[i, k] = 1

    def update_center(self):
        """
        update clustering center mu_k
        :return:
        """
        # for k in range(self.cluster_num):
        #     gamma_k = self.gamma[:, k]
        #     samples_in_k = np.dot(gamma_k, self.samples)
        #     mu_k = samples_in_k / np.sum(gamma_k)
        #     self.center[k, :] = mu_k
        gamma_sample_sum = np.dot(self.gamma.T, self.samples)
        gamma_sum = np.sum(self.gamma, axis=0)
        # RuntimeWarning: invalid value encountered in divide
        epsilon = 1e-10
        self.center = np.where(gamma_sum[:, np.newaxis] > 0, gamma_sample_sum / (gamma_sum[:, np.newaxis]+ epsilon), self.center)

    def predict_for_train_data(self):
        # use argmax to find k=1
        predicts = np.argmax(self.gamma, axis=1)
        return predicts

    def predict_for_test_data(self, test_samples):
        predicts = np.zeros(test_samples.shape[0])
        for i in range(test_samples.shape[0]):
            sample_i = test_samples[i]
            distances = np.linalg.norm(self.center - sample_i, axis=1)
            k = np.argmin(distances)
            predicts[i] = k
        return predicts

    def train(self):
        self.init_center()
        with tqdm(range(self.iters), desc="K-Means Training Progress") as pbar:
            for i in pbar:
                # early stop if CONVERGE
                prev_center = self.center.copy()
                self.update_cluster()
                self.update_center()
                # print(f'--finish iter-{i} traning')
                # if ord == Inf:
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                # or, Solution: use ord = 'fro', caculate Matrix's Frobenius Norm
                convergence = np.linalg.norm(prev_center - self.center, ord='fro')
                pbar.set_postfix({"Iter": i, "Convergence": convergence})
                if convergence < 1e-5:
                    print("CENTERS ALREADY CONVERGED")
                    break

    def get_centers(self):
        return self.center

