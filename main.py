import sys
import numpy as np
from typing import Tuple

from src.train import read_transfomer, set_langs, train
from src.train import set_langs
from src.train import save_transformer_summary
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
    return int(true_expansion == pred_expansion)



def predict(f: str, transformer):
    """ the prediction function - it receives a transformer (model used) and """
    return  translate(transformer, f)



def main(filepath: str, argv):
    factors, expansions = load_file(filepath)
    N = len(factors)
    print('N: ', N)
    threshold = int(N*float(argv[1]))
    print('Threshold: ', threshold)

    set_langs(factors, expansions)

    if "train" in argv:
        print('Train mode.')
        train_factors = factors[:threshold]
        train_expansions = expansions[:threshold]

        train(train_factors, train_expansions, NUM_EPOCHS=16)
    elif "summary" in argv:
        print('Saving summary...') 
        save_transformer_summary(factors[0:1], expansions[0:1])
    else:
        print('Test mode.')
        factors = factors[threshold:]
        expansions = expansions[threshold:]
        N = len(factors)
        transformer = read_transfomer()
        pred = []
        for i, f in enumerate(factors): 
            if i % 200 == 0:
                print(f"Translating factor expression %d/%d."%(i, N))
            pred.append(translate(transformer, f))
     
        scores = [score(te, pe) for te, pe in zip(expansions, pred)]
        print(np.mean(scores))   

     

if __name__ == "__main__":
    print("Starting...")
    main("test.txt" if "-t" in sys.argv else "train.txt", sys.argv)
    print("Done.")
    sys.exit()
    
