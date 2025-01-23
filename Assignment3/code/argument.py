import argparse


def args():
    parser = argparse.ArgumentParser(description="arguments for lab 3")
    parser.add_argument("--model", type=str, default="GMM", help="GMM or KMeans")
    parser.add_argument("--init_method", type=str, default="KMeans-PreTrain", help="random or dist-based or KMeans-PreTrain")
    parser.add_argument("--covar_type", type=str, default="full", help="full or tied or diag or spherical")
    return parser.parse_args()
