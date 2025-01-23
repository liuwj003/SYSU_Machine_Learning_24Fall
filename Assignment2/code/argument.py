import argparse


def args():
    parser = argparse.ArgumentParser(description="arguments for lab 2")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model", type=str, default="CNN", help="Softmax or MLP or CNN")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size for training")
    parser.add_argument("--epochs", type=int, default=30, help="epochs for training")
    parser.add_argument("--optimizer", type=str, default="Adam", help="SGD or Momentum or Adam")
    return parser.parse_args()
