from argument import args
from preprocess import extract_data, standardization, min_max_normalization
from LinearClassify_model import LinearClassifier
from SVM_model import SVM


def main():
    arguments = args()
    train_labels, train_samples_before = extract_data('mnist_01_train.csv')
    train_samples = standardization(train_samples_before)
    # train_samples = min_max_normalization(train_samples_before)
    test_labels, test_samples_before = extract_data('mnist_01_test.csv')
    test_samples = standardization(test_samples_before)
    # test_samples = min_max_normalization(test_samples_before)

    if arguments.model == "linear":
        model = LinearClassifier(arguments.loss_func, train_samples, train_labels, arguments.learning_rate, arguments.epochs, arguments.regular_strength, arguments.batch_size)
        model.para_initialize()
        model.train()
        model.evaluate(test_samples, test_labels)
        model.plot_loss()
    elif arguments.model == "SVM":
        model = SVM(arguments.kernel_func, train_samples, train_labels)
        model.train()
        model.evaluate(test_samples, test_labels)


if __name__ == '__main__':
    main()
