import argparse


def args():
    parser = argparse.ArgumentParser(description="arguments for lab 1")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--model", type=str, default="linear", help="choose SVM_model or LinearClassifier_model")
    parser.add_argument("--loss_func", type=str, default="cross-entropy", help="choose hinge loss or cross-entropy loss")
    parser.add_argument("--kernel_func", type=str, default="Gaussian", help="choose Gaussian kernel or linear kernel")
    parser.add_argument("--regular_strength", type=float, default=1e-3, help="regularization strength")
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs for the model")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")

    return parser.parse_args()
