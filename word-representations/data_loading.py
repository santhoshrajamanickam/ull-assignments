import numpy as np

def load_embeddings(file_path):
    """
    Loads word embeddings from a text file and returns a dictionary
    of embeddings.
    Args:
        - file_path (str): the path to the text fileself.
    Returns:
        - dict, mapping word (str) to numpy array containing embedding.
    """
    embeddings = {}
    with open(file_path) as file:
        for line in file:
            values = line.strip().split()
            embeddings[values[0]] = np.array(values[1:]).astype(np.float)
    return embeddings

def load_sim_dataset(file_path, score_col, skip=0):
    """
    Loads a similarity dataset and returns a dictionary of pairs to scores.
    Args:
        - file_path (str): the path to the dataset.
        - score_col (int): the column in the file with the score of the word pair.
        - skip (int): number of lines to skip at the beginning of the file.
    Returns:
        - pairs (list): contains pairs of words (tuple of str)
        - scores (list): contains scores for pairs (float)
    """
    pairs = []
    scores = []
    with open(file_path) as file:
        for i in range(skip): file.readline()
        for line in file:
             values = line.split()
             word1 = values[0]
             word2 = values[1]
             score = float(values[score_col])
             pairs.append((word1, word2))
             scores.append(score)
    return pairs, scores
