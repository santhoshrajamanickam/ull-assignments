import numpy as np

class Embeddings:
    def __init__(self, file_path):
        """
        Loads word embeddings from a text file and returns a dictionary
        of embeddings.
        Args:
            - file_path (str): the path to the text file.
        Returns:
            - dict, mapping word (str) to numpy array containing embedding.
        """
        self.word2idx = {}
        self.idx2word = {}
        embeddings = []
        with open(file_path) as file:
            for i, line in enumerate(file):
                values = line.strip().split()
                word = values[0]
                self.word2idx[word] = i
                self.idx2word[i] = word
                embeddings.append(np.array(values[1:]).astype(np.float))
                #if i == 100: break

        self.vectors = np.array(embeddings)
        self.norms = np.linalg.norm(self.vectors, axis=1)
        self.norm_vectors = self.vectors/self.norms[:, np.newaxis]

    def _cosine_similarity(self, vec1, norm1, vec2, norm2):
        return ((vec1 @ vec2)/norm1)/norm2

    def similarity(self, word, word2=None):
        """
        Returns the cosine similarity between words.
        Args:
            - word (str): first word to compare.
            - word2 (str): optional, second word to compare. If None,
                the similarities with all words is returned.
        Returns:
            - numpy array containing similarities.
        """
        # Get word embedding and norm
        word_idx = self.word2idx[word]
        word_emb = self.vectors[word_idx, :]
        emb_norm = self.norms[word_idx]

        if word2 is None:
            # Calculate similarity with all available words
            #return ((self.vectors @ word_emb)/self.norms)/emb_norm
            return self._cosine_similarity(self.vectors, self.norms, word_emb, emb_norm)
        else:
            # Calculate similarity between word pairs
            word_idx2 = self.word2idx[word2]
            word_emb2 = self.vectors[word_idx2, :]
            emb_norm2 = self.norms[word_idx2]
            #return (word_emb @ word_emb2)/(emb_norm * emb_norm2)
            return self._cosine_similarity(word_emb, emb_norm, word_emb2, emb_norm2)

    def top_similar(self, word, n=5):
        """
        Returns the top n similar words for the given word.
        Args:
            - word (str).
            - n (int): optional, number of similar words to return.
        Returns:
            list of str containing top n words.
        """
        if n - 1 > len(self.word2idx):
            raise ValueError("n is larger than the number of embeddings.")

        # Calculate similarity with all available words
        similarities = self.similarity(word)
        # Return top n similar words
        return [self.idx2word[i] for i in np.argsort(similarities)][-2:-2-n:-1]

    def analogy(self, a, b, c):
        """
        Returns the answer to 'a is to b, as words are to ?'.
        The answer is the word whose word embedding is closer to the result of
        calculating b - a + c using their respective word embeddings.
        """
        # Compute result embedding
        a_idx = self.word2idx[a]
        b_idx = self.word2idx[b]
        c_idx = self.word2idx[c]
        d_emb = self.norm_vectors[b_idx] - self.norm_vectors[a_idx] + self.norm_vectors[c_idx]
        d_norm = np.linalg.norm(d_emb)
        # Get similarities with all embeddings
        similarities = self._cosine_similarity(self.vectors, self.norms, d_emb, d_norm)
        # Get word indices sorted by similarity
        sorted_idx = np.argsort(similarities)
        # Return sorted words without including input words
        return [self.idx2word[idx] for idx in sorted_idx[::-1] if idx not in (a_idx, b_idx, c_idx)]

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

def load_analogy_dataset(file_path):
    """
    Loads an analogy dataset containing 4 words per line.
    Lines starting with ':' are omitted.
    Args:
        - file_path (str): the path to the dataset.
    Returns:
        - list of list, each containing 4 words (str).
    """
    with open(file_path) as file:
        analogies = []
        for line in file:
            if line[0] != ':':
                (a, b, c, d) = line.lower().split()
                analogies.append((a, b, c, d))
    return analogies
