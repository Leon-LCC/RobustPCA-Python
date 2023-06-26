import numpy as np
import pandas as pd
import argparse
import tqdm

from RobustPCA import RPCA, RPCA_inexact



def main(args):
    # Load data
    train = np.load(args.i)
    label = np.load(args.g)

    # Transpose and Normalize
    train = train.T / 255

    # Parameters
    l = args.l
    rho = args.r
    t = args.t

    # Robust PCA
    A, E = RPCA_inexact(train, l, 1.25 / np.linalg.norm(train, ord=2, axis=(0,1)), rho)

    # Filter
    E = np.where(E > t, E, 0)

    # Anomaly if any pixel is not zero
    result = np.where(np.sum(E, axis=0) > 0, 1, 0)

    # Accuracy
    accuracy = np.sum(result == label) / len(label)
    print("Accuracy: ", accuracy)

    # Save result
    df = pd.DataFrame({'id': np.arange(len(result)), 'category': result})
    df.to_csv(args.o, index=False, header=['id', 'category'])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Part2: Anomaly Detection')
    parser.add_argument('--i', '--input', type=str, required=True, help='input data path')
    parser.add_argument('--g', '--groundtruth', type=str, required=True, help='groundtruth path')
    parser.add_argument('--o','--output', type=str, required=True, help='output csv path')
    parser.add_argument('--l', '--lambda', type=float, default=0.053, help='lambda')
    parser.add_argument('--r', '--rho', type=float, default=1.5, help='rho')
    parser.add_argument('--t', '--threshold', type=float, default=0.999999, help='threshold')
    main(parser.parse_args())