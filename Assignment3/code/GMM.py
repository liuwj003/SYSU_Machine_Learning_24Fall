import numpy as np
from KMeans import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm
from utils import distance, far, furthest


class GMM:
    def __init__(self, init_method, covar_type, cluster_num, samples, iters):
        """
        Gaussian Mixture Model
        :param init_method: random or KMeans-PreTrain
        :param covar_type: type of covariance matrix, full or tied or diag or spherical
        :param cluster_num: 10 in this lab
        :param samples: samples (N=60000, D)
        """
        self.init_method = init_method
        self.covariance_type = covar_type
        self.cluster_num = cluster_num
        self.samples = samples
        self.samples_num = samples.shape[0]
        self.feature_dim = samples.shape[1]
        self.iters = iters

        # parameters of GMM model
        self.means = np.zeros((self.cluster_num, self.feature_dim))  # μ_k, center of k-th cluster
        # "tied" has different covariances-shape from other three, so we just set it None here
        self.covariances = None  # Σ_k: covariance matrices
        self.weights = np.ones(self.cluster_num) / self.cluster_num  # π_k: cluster weights
        self.resp = np.zeros((self.samples_num, self.cluster_num))  # γ_nk

    def init_parameters(self):
        """
        initialize GMM model parameters: μ_k, Σ_k, π_k
        """
        # initialize means
        if self.init_method == "random":
            random_indices = np.random.choice(self.samples_num, size=self.cluster_num, replace=False)
            self.means = self.samples[random_indices, :]

        elif self.init_method == "KMeans-PreTrain":
            kmeans = KMeans("dist-based", 10, self.samples, 30)
            kmeans.train()
            self.means = kmeans.get_centers()

        else:
            raise ValueError("Invalid means-init-method")

        # initialize covariances
        epsilon = 1e-3  # to avoid singular matrix
        if self.covariance_type == "spherical":
            unit_matrix = np.eye(self.feature_dim)  # generate unit matrix I
            # haven't any cluster yet, so we set all sigma_k the same as var(all samples)
            squared_sigma_k = np.var(self.samples)
            cov_k = squared_sigma_k * unit_matrix
            # print(cov_k)
            # generate cov_matrix of cluster_num
            self.covariances = np.array([cov_k for _ in range(self.cluster_num)])

        elif self.covariance_type == "diag":
            squared_sigma_k_for_each_dim = np.var(self.samples, axis=0) + epsilon
            cov_k = np.diag(squared_sigma_k_for_each_dim)
            # print(cov_k[:, 300])
            self.covariances = np.array([cov_k for _ in range(self.cluster_num)])

        elif self.covariance_type == "full":
            cov_k = np.cov(self.samples.T) + np.eye(self.feature_dim) * epsilon
            # print(cov_k[:, 300])
            self.covariances = np.array([cov_k for _ in range(self.cluster_num)])

        elif self.covariance_type == "tied":
            self.covariances = np.cov(self.samples.T) + np.eye(self.feature_dim) * epsilon

        else:
            raise ValueError("Invalid covariance_type")

        # initialize weights
        # same as __init__: Initially, we set all weights have same value


    def e_step(self):
        """
        compute the responsibility γ_nk: the probability of sample_n belongs to k-th cluster
        """
        # clear previous result
        self.resp = np.zeros((self.samples_num, self.cluster_num))

        for k in range(self.cluster_num):
            if self.covariance_type in ["spherical", "diag", "full"]:
                cov_matrix_k = self.covariances[k]
            elif self.covariance_type == "tied":
                cov_matrix_k = self.covariances
            else:
                raise ValueError("Invalid covariance_type")

            gaussian = multivariate_normal(mean=self.means[k], cov=cov_matrix_k)
            self.resp[:, k] = self.weights[k] * gaussian.pdf(self.samples)

        self.resp = self.resp / np.sum(self.resp, axis=1, keepdims=True)


    def m_step(self):
        """
        update μ_k, Σ_k, π_k (means, covariances, weights)
        """
        N_k = np.sum(self.resp, axis=0)
        # update π_k
        self.weights = N_k / self.samples_num

        # update μ_k
        for k in range(self.cluster_num):
            self.means[k, :] = np.dot(self.resp[:, k], self.samples) / N_k[k]

        # update Σ_k
        if self.covariance_type == "full":
            for k in range(self.cluster_num):
                diff = self.samples - self.means[k]
                # resp[:, k]: (N,)一维向量
                # 这里要进行逐元素加权操作，用[:, np.newaxis]变成 N*1 的列向量
                # 这样，如果每行是1，那么对应的 diff 行（第n个样本与第k个中心的差）才会被加权1，否则加权0
                gamma_weighted_diff = self.resp[:, k][:, np.newaxis] * diff
                # 这里注意一下，样本是按行存储的，和手动数学计算（按列存储）不同，所以转秩不一样
                self.covariances[k, :] = np.dot(gamma_weighted_diff.T, diff) / N_k[k]

        elif self.covariance_type == "diag":
            for k in range(self.cluster_num):
                diff = self.samples - self.means[k]
                gamma_weighted_diff = self.resp[:, k][:, np.newaxis] * (diff ** 2)
                # 记得把向量还原成对角矩阵（np.diag()）
                self.covariances[k, :] = np.diag(np.sum(gamma_weighted_diff, axis=0) / N_k[k])

        elif self.covariance_type == "spherical":
            for k in range(self.cluster_num):
                diff = self.samples - self.means[k]
                squared_l2_norm = np.sum(diff ** 2, axis=1)
                squared_sigma_k = np.sum(self.resp[:, k] * squared_l2_norm) / (N_k[k] * self.feature_dim)
                self.covariances[k, :] =  np.eye(self.feature_dim) * squared_sigma_k

        elif self.covariance_type == "tied":
            cov_sum = np.zeros((self.feature_dim, self.feature_dim))
            for k in range(self.cluster_num):
                diff = self.samples - self.means[k]
                gamma_weighted_diff = self.resp[:, k][:, np.newaxis] * diff
                cov_sum += np.dot(gamma_weighted_diff.T, diff)

            self.covariances = cov_sum / self.samples_num

        else:
            raise ValueError("Invalid covariance_type")


    def predict_for_train_data(self):
        predicts = np.argmax(self.resp, axis=1)
        return predicts

    def predict_for_test_data(self, test_samples):
        predicts = np.zeros(test_samples.shape[0])
        predict_resp = np.zeros((test_samples.shape[0], self.cluster_num))
        
        # similar to e-step
        for k in range(self.cluster_num):
            if self.covariance_type in ["spherical", "diag", "full"]:
                cov_matrix_k = self.covariances[k]

            elif self.covariance_type == "tied":
                cov_matrix_k = self.covariances

            else:
                raise ValueError("Invalid covariance_type")

            # create a random variable ~ our gaussian distribution
            random_multi_gaussian = multivariate_normal(mean=self.means[k], cov=cov_matrix_k)
            # input sample into the pdf, get the probability result
            prob_k = random_multi_gaussian.pdf(test_samples)
            predict_resp[:, k] = self.weights[k] * prob_k

        row_sums = np.sum(predict_resp, axis=1, keepdims=True)
        predict_resp = predict_resp / row_sums
        predicts = np.argmax(predict_resp, axis=1)

        return predicts


    def train(self):
        self.init_parameters()
        with tqdm(range(self.iters), desc="EM Training Progress") as pbar:
            for i in pbar:
                prev_means = self.means.copy()
                # print("e-step")
                self.e_step()
                # print("m-step")
                self.m_step()
                # print(f'--finish iter-{i} EM traning')
                pbar.set_postfix({"Iter": i, "Convergence": np.linalg.norm(prev_means - self.means, ord='fro')})
                if np.linalg.norm(prev_means - self.means, ord='fro') < 1e-5:
                    print("CENTERS ALREADY CONVERGED")
                    break

    def get_means(self):
        return self.means



