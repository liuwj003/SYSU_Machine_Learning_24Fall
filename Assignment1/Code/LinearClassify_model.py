import numpy as np
from loss_function import sigmoid, hinge_loss, grad_hinge, cross_entropy_loss, grad_cross_entropy
import matplotlib.pyplot as plt
from preprocess import data_shuffle


class LinearClassifier:
    """
    Linear Classifier using hinge loss or cross-entropy loss
    """
    def __init__(self, loss_func, samples, labels, lr, epochs, regu_strength, batch_size):
        self.loss_func = loss_func
        self.samples = samples
        if loss_func == "hinge":
            self.labels = np.where(labels == 0, -1, labels)
        elif loss_func == "cross-entropy":
            self.labels = labels
        else:
            raise ValueError("Invalid Loss Function")
        self.lr = lr
        self.epochs = epochs
        self.regu_strength = regu_strength
        self.batch_size = batch_size
        self.weights = np.zeros(samples.shape[1])
        self.bias = 0.0  # float
        self.training_loss = []  # for future drawing

    def para_initialize(self):
        """
        initialize weights and bias
        """
        np.random.seed(4219)
        self.weights = np.random.randn(self.samples.shape[1]) * 0.01  # normal distribution N(0.0.01^2)
        self.bias = 0.0  # still 0.0

    def para_update(self, batch_labels, batch_samples):
        """
        update weights and bias with gradient descendant method
        """
        if self.loss_func == "hinge":
            # grad_w, grad_b = grad_hinge(self.labels, self.samples, self.weights, self.bias, self.regu_strength)
            grad_w, grad_b = grad_hinge(batch_labels, batch_samples, self.weights, self.bias, self.regu_strength)
        elif self.loss_func == "cross-entropy":
            # grad_w, grad_b = grad_cross_entropy(self.labels, self.samples, self.weights, self.bias, self.regu_strength)
            grad_w, grad_b = grad_cross_entropy(batch_labels, batch_samples, self.weights, self.bias, self.regu_strength)
        else:
            raise ValueError("Invalid Loss Function")
        # then update
        self.weights = self.weights - self.lr * grad_w
        self.bias = self.bias - self.lr * grad_b

    def curr_loss(self):
        """
        :return: current loss of the model
        """
        if self.loss_func == "hinge":
            loss_result = hinge_loss(self.labels, self.samples, self.weights, self.bias, self.regu_strength)
        elif self.loss_func == "cross-entropy":
            loss_result = cross_entropy_loss(self.labels, self.samples, self.weights, self.bias, self.regu_strength)
        else:
            raise ValueError("Invalid Loss Function")
        return loss_result

    def hinge_predict(self, input_samples):
        """predict result in {-1, 1}"""
        predict = np.sign(np.dot(self.weights, input_samples.T) + self.bias)
        return predict

    def sigmoid_predict(self, input_samples):
        """predict result in {0, 1}"""
        predict = sigmoid(np.dot(self.weights, input_samples.T) + self.bias)
        result = np.where(predict >= 0.5, 1, 0)
        return result

    def train(self):
        """
        train our model with weights already initialized, data already normalized
        using mini-batch
        """
        for epoch in range(self.epochs + 1):
            # firstly shuffle our data
            shuffled_samples, shuffled_labels = data_shuffle(self.samples, self.labels)

            # then traverse every batch to update
            for j in range(0, len(self.labels), self.batch_size):
                batch_samples = shuffled_samples[j: min(j+self.batch_size, len(self.labels))]
                batch_labels = shuffled_labels[j: min(j+self.batch_size, len(self.labels))]
                self.para_update(batch_labels, batch_samples)

            if epoch % 10 == 0:
                curr_loss = self.curr_loss()
                self.training_loss.append(curr_loss)
                if epoch % 20 == 0:
                    print(f"Epochs {epoch}, Loss: {curr_loss}")

    def predict(self, test_samples):
        """predict result in {0, 1}"""
        if self.loss_func == "hinge":
            predict_0 = self.hinge_predict(test_samples)
            predict = np.where(predict_0 == -1, 0, 1)
        elif self.loss_func == "cross-entropy":
            predict = self.sigmoid_predict(test_samples)
        else:
            raise ValueError("Invalid Loss Function")
        return predict

    def evaluate(self, test_samples, test_labels):
        """
        evaluate accuracy of our model on testing dataset
        """
        test_predict = self.predict(test_samples)
        accuracy = np.mean(test_predict == test_labels)
        print(f"Accuracy: {accuracy:.6%}")
        return accuracy

    def plot_loss(self):
        """plot training loss curve"""
        epochs_points = [i * 10 for i in range(len(self.training_loss))]
        plt.figure()
        plt.plot(epochs_points, self.training_loss, label="Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid()
        plt.legend()
        save_path = f"linear_{self.loss_func}_training_loss.png"
        plt.savefig(save_path, format="png")
        plt.show()

    def visualize(self, test_samples, test_labels):
        pass
