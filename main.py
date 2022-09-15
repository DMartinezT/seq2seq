import sys
import numpy as np
from typing import Tuple

from src.train import read_transfomer, set_langs, train
from src.train import set_langs
from src.train import read_aux
from src.train import translate




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
    print(true_expansion, "!!!!!!!!!!!!!!", pred_expansion)
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(f: str, transformer):

    return  translate(transformer, f)


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str, test = True, p = 0.7):
    factors, expansions = load_file(filepath)
    N = len(factors)
    print('N: ', N)
    threshold = int(N*float(p))
    print('Threshold: ', threshold)
    set_langs(factors, expansions)
    print('test: ', int(test))
    if int(test):
        print('Test mode.')
        factors = factors[threshold+1:]
        expansions = expansions[threshold+1:]

        transformer = read_transfomer()
        pred = [predict(f, transformer) for f in factors]
        scores = [score(te, pe) for te, pe in zip(expansions, pred)]
        print(np.mean(scores))
    
    else:
        print('Train mode.')
        factors = factors[:threshold]
        expansions = expansions[:threshold]

        train(factors, expansions, 0.8, 10)

def plot_main(filepath: str):
    print('Plotting happily!!')
    factors, expansions = load_file(filepath)
    set_langs(factors, expansions)
    N = len(factors) 
    read_aux(factors[N-5:N-1], expansions[N-5:N-1])



if __name__ == "__main__":
    print("Starting...")
    if sys.argv[1] == str(5):
        plot_main("train.txt")
    else:
        main("test.txt" if "-t" in sys.argv else "train.txt", sys.argv[1], sys.argv[2])
    print("Done.")
    sys.exit()
    
