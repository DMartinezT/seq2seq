import sys
import numpy as np
from typing import Tuple

from src.train import set_langs, train
from src.train import read
from src.train import set_langs




MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    
    read(factors)

    return factors


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str, test = True, p = 0.7):
    factors, expansions = load_file(filepath)
    N = len(factors)
    threshold = int(N*float(p))
    set_langs(factors, expansions)
    print('test: ', int(test))
    if int(test):
        print('Test mode.')
        factors = factors[threshold+1:]
        expansions = expansions[threshold+1:]

        pred = [predict(f) for f in factors]
        scores = [score(te, pe) for te, pe in zip(expansions, pred)]
        print(np.mean(scores))
    
    else:
        print('Train mode.')
        factors = factors[:threshold]
        expansions = expansions[:threshold]

        train(factors, expansions, 0.8, 10)


if __name__ == "__main__":
    print("Starting...")
    main("test.txt" if "-t" in sys.argv else "train.txt", sys.argv[1], sys.argv[2])
    print("Done.")
    sys.exit()
    
