from argument import args
from preprocess import extract_data, standardization
from KMeans import KMeans
from GMM import GMM
from utils import calculate_acc
from sklearn.decomposition import PCA
import numpy as np


def main():
    arguments = args()
    train_labels, train_samples_before = extract_data('mnist_train.csv')
    # train_samples = standardization(train_samples)
    test_labels, test_samples_before = extract_data('mnist_test.csv')
    # test_samples = standardization(test_samples)
    pca = PCA(50)
    pca.fit(train_samples_before)
    train_samples = pca.transform(train_samples_before)
    test_samples = pca.transform(test_samples_before)
    # train_samples = train_samples_before
    # test_samples = test_samples_before

    if arguments.model == "GMM":
        gmm = GMM(arguments.init_method, arguments.covar_type, 10, train_samples, 300)
        gmm.train()
        # result = gmm.get_means()
        predicts = gmm.predict_for_test_data(test_samples)
    elif arguments.model == "KMeans":
        kmeans = KMeans(arguments.init_method, 10, train_samples, 500)
        kmeans.train()
        # result = kmeans.get_centers()
        predicts = kmeans.predict_for_test_data(test_samples)
    else:
        raise ValueError("Invalid model name")

    acc = calculate_acc(test_labels, predicts)
    if arguments.model == "GMM":
        print(f'Clustering Accuracy for Model {arguments.model} with Init-Method {arguments.init_method}, Cov {arguments.covar_type}: {acc}')
    elif arguments.model == "KMeans":
        print(f'Clustering Accuracy for Model {arguments.model} with Init-Method {arguments.init_method}: {acc}')

if __name__ == '__main__':
    main()
