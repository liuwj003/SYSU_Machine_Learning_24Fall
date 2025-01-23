import numpy as np
from sklearn.svm import SVC


class SVM:
    def __init__(self, kernel, train_samples, train_labels):
        if kernel == "Gaussian":
            self.kernel = "rbf"
        elif kernel == "Linear":
            self.kernel = "linear"
        self.train_samples = train_samples
        self.train_labels = train_labels
        self.clf = SVC(kernel=self.kernel)

    def train(self):
        self.clf.fit(self.train_samples, self.train_labels)

    def predict(self, test_samples):
        """ return type is 'numpy.ndarray' """
        return self.clf.predict(test_samples)

    def evaluate(self, test_samples, test_labels):
        predictions = self.predict(test_samples)
        accuracy = np.mean(predictions == test_labels)
        print(f"Accuracy: {accuracy:.6%}")
        return accuracy
