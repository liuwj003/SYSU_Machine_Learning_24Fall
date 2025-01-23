import argparse


def args():
    parser = argparse.ArgumentParser(description="arguments for lab 4")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=30, help="Latent variable dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size for training")
    parser.add_argument("--epochs", type=int, default=200, help="epochs for training")
    return parser.parse_args()
